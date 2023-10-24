import asyncio
import itertools
import json
import os
import tempfile
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsonlines
import petname
import tiktoken

from absa_llm.models import MODELS
from absa_llm.processor import process_api_requests_from_file


@dataclass
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


@dataclass
class Example:
    context: str
    completion: str

    def to_chat_messages(self) -> List[ChatMessage]:
        return [
            ChatMessage(role="user", content=self.context),
            ChatMessage(role="system", content=self.completion),
        ]

    def to_dict(self) -> dict[str, Any]:
        return {"context": self.context, "completion": self.completion}


@dataclass
class IdTextPair:
    id: str
    text: str


@dataclass
class Params:
    model: str
    temperature: float
    max_tokens: int
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    system_message: str
    system_message_name: str
    logit_bias_literal: Optional[Dict[str, float]] = None
    logit_bias: Optional[Dict[int, float]] = None
    train_examples: List[Example] = field(default_factory=list)
    examples_in_separate_messages: bool = False
    use_function_calling: bool = True

    def __post_init__(self):
        if self.logit_bias_literal is not None:
            self.logit_bias = self.encode_logit_bias(self.logit_bias_literal)

    @staticmethod
    def encode_logit_bias(logit_bias: dict[str, float]) -> dict[int, float]:
        """Encodes the logit bias using the TikTok tokenizer."""
        encoding = tiktoken.get_encoding("cl100k_base")
        bias = {encoding.encode(k)[0]: v for k, v in logit_bias.items()}
        return bias

    def to_dict(self) -> dict[str, Any]:
        """
        Returns a dictionary representation of the parameters.
        Only parameters used by ChatCompletion are included.
        """
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "logit_bias": self.logit_bias,
        }


def count_tokens(text: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(text))
    return num_tokens


def hash_dict(d: dict) -> str:
    """Return a hash of a dictionary."""
    return sha256(json.dumps(d).encode("utf-8")).hexdigest()


class ChatCompletionJob:
    def __init__(
        self,
        responses_path: Path,
        id_text_pairs: List[IdTextPair],
        params: Params,
        function: Optional[Dict[Any, Any]],
        max_attempts: int = 10,
    ):
        """
        Manages runs of ChatCompletions.
        - responses_path: Path to the file where the responses will be stored.
        - params: Parameters for the ChatCompletion.
        - id_text_pairs: The IDs and texts to be processed.
        - function: A dictionary describing the function to be used. If a
          function is provided, the ChatCompletion will be forced to use it.
          Don't use a function with finetuned models.
        - max_attempts: The maximum number of retries for a failed request.
        """
        self.responses_path = responses_path
        self.id_text_pairs = id_text_pairs
        self.params = params
        self.function = function
        self.max_attempts = max_attempts

        if self.params.model not in MODELS:
            raise ValueError("Model not supported.")

        # Generate a unique but human readable ID for this run
        self.run_id = petname.generate()

        # Check that the OPENAI_API_KEY environment variable is set
        if os.getenv("OPENAI_API_KEY") is None:
            raise ValueError("OPENAI_API_KEY environment variable not set.")

    def __len__(self):
        """Return the number of texts."""
        return len(self.id_text_pairs)

    def prompt_tokens(self):
        """
        Counts the tokens in the prompts which in include system message,
        training example, texts to be processed and the function.
        """

        # Count tokens for the parts that are repeated for each text
        system_message_tokens = count_tokens(self.params.system_message)

        example_tokens = sum(
            [
                count_tokens(example.context) + count_tokens(example.completion)
                for example in self.params.train_examples
            ]
        )

        if self.function is None:
            function_tokens = 0
        else:
            function_tokens = count_tokens(json.dumps(self.function))

        repeated_tokens = system_message_tokens + example_tokens + function_tokens

        # Count tokens for the reviews
        text_tokens = sum([count_tokens(t.text) for t in self.id_text_pairs])

        prompt_tokens = repeated_tokens * len(self.id_text_pairs) + text_tokens

        return prompt_tokens

    def estimate_cost(self) -> float:
        """
        Estimates the cost of executing the ChatCompletion based on token
        counts and model prices.
        """
        prompt_tokens = self.prompt_tokens()
        completion_tokens_max = self.params.max_tokens * len(self.id_text_pairs)

        model = self.params.model
        price_prompt_tokens_1k = MODELS[model]["price_prompt_tokens_1k"]
        price_completion_tokens_1k = MODELS[model]["price_completion_tokens_1k"]

        prompt_cost = prompt_tokens * price_prompt_tokens_1k / 1000
        completion_cost_max = completion_tokens_max * price_completion_tokens_1k / 1000

        total_cost_max = prompt_cost + completion_cost_max

        return total_cost_max

    def build_messages(
        self,
        text: str,
    ) -> List[ChatMessage]:
        "Prepare a series of chat messages for a ChatCompletion prompt."

        if self.params.system_message != "":
            chat = [
                ChatMessage(role="system", content=self.params.system_message),
            ]
        else:
            chat = []

            if (
                not self.params.examples_in_separate_messages and
                len(self.params.train_examples) > 0
            ):
                raise ValueError(
                    "If examples are to be included in the same message as the "
                    "system message, a non-empty system message must be provided."
                )

        if self.params.examples_in_separate_messages:
            # Add chat messages for each training example, alternating between
            # user and assistant
            train_chat_messages = [
                example.to_chat_messages() for example in self.params.train_examples
            ]

            chat.extend(itertools.chain(*train_chat_messages))

        else:
            # Add all training examples in the system message
            train_string = "\n".join(
                [
                    example.context + "\n" + example.completion
                    for example in self.params.train_examples
                ]
            )

            if train_string != "":
                chat[0].content += "\n"
                chat[0].content += train_string

        chat.append(ChatMessage(role="user", content=text))

        return chat

    def build_requests(self) -> None:
        """
        Build a list of requests to be sent to the OpenAI API.
        """
        # Build chat prompts
        chats = [self.build_messages(text=t.text) for t in self.id_text_pairs]

        # Build full requests
        requests = [
            {
                "model": self.params.model,
                "messages": [m.to_dict() for m in chat],
                "temperature": self.params.temperature,
                "max_tokens": self.params.max_tokens,
                "top_p": self.params.top_p,
                "frequency_penalty": self.params.frequency_penalty,
                "presence_penalty": self.params.presence_penalty,
                "logit_bias": self.params.logit_bias,
            }
            for chat in chats
        ]

         # Force model to call the function if a function is provided
        if self.function is not None:
            function_call = {"name": self.function["name"]}
            functions = [self.function]

            for request in requests:
                request["functions"] = functions
                request["function_call"] = function_call

        self.requests = requests

    def get_completions(
        self,
        request_file_path: Optional[Path] = None,
        response_file_path: Optional[Path] = None,
    ) -> None:
        """
        Send texts to OpenAI API and return the results.

        Args:
            request_file_path: Path to save the requests JSONL to.
                Defaults to None. In this case, a temporary file will be used.
            response_file_path: Path to save the responses JSONL to.
                Defaults to None. In this case, a temporary file will be used.
        """

        if not hasattr(self, "requests"):
            self.build_requests()

        # Create a hash map of the requests to sort the results later
        request_order = {
            hash_dict(request): i for i, request in enumerate(self.requests)
        }

        # Define input and output filepaths
        if request_file_path is None:
            requests_file = tempfile.NamedTemporaryFile(suffix=".jsonl")
            requests_file_name = requests_file.name
        else:
            requests_file_name = request_file_path

        if response_file_path is None:
            save_file = tempfile.NamedTemporaryFile(suffix=".jsonl")
            save_file_name = save_file.name
        else:
            save_file_name = response_file_path

        # Write requests to input file
        with jsonlines.open(requests_file_name, "w") as writer:
            for request in self.requests:
                writer.write(request)

        # Process requests
        safety_factor = 0.6  # as recommended by OpenAI

        asyncio.run(
            process_api_requests_from_file(
                requests_filepath=str(requests_file_name),
                save_filepath=str(save_file_name),
                request_url="https://api.openai.com/v1/chat/completions",
                api_key=os.environ["OPENAI_API_KEY"],
                max_requests_per_minute=MODELS[self.params.model][
                    "max_requests_per_minute"
                ]
                * safety_factor,
                max_tokens_per_minute=MODELS[self.params.model]["max_tokens_per_minute"]
                * safety_factor,
                token_encoding_name="cl100k_base",
                max_attempts=self.max_attempts,
                logging_level=20,
            )
        )

        # Results are not sorted due to asynchronous processing
        with jsonlines.open(save_file_name, "r") as reader:
            results = [r for r in reader]

        assert len(results) == len(self.requests), "Wrong number of results."

        # Sort results in the same order as the requests
        # hash the first list element (the request) and use the hash as a key
        results = sorted(
            results,
            key=lambda x: request_order[hash_dict(x[0])],
        )

        self.results = results

    def save_completions(self) -> None:
        "Append the completions to the output file."

        if not hasattr(self, "results"):
            raise ValueError("No results to save.")

        with jsonlines.open(self.responses_path, mode="a") as writer:
            for i, completion in enumerate(self.results):
                # Merge the input and output dicts to get 1 dict per response
                completion = {**completion[0], **completion[1]}
                completion.update(
                    {
                        "params": self.params.to_dict(),
                        "id": self.id_text_pairs[i].id,
                        "run_id": self.run_id,
                        "system_message": self.params.system_message,
                        "system_message_name": self.params.system_message_name,
                        "n_train_examples": len(self.params.train_examples),
                        "examples_in_separate_messages": self.params.examples_in_separate_messages,
                        "function": self.function,
                    }
                )
                writer.write(completion)
