# %%
import gzip
import json
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Optional

import boto3
import pandas as pd
import xmltodict

from absa_llm.job import Example


# %%
def read_semeval_xml(filepath: Path) -> pd.DataFrame:
    """Read the SemEval 2014 XML files into a DataFrame."""
    with open(filepath, "rb") as f:
        parsed_dict = xmltodict.parse(f.read())  # easier than pd.read_xml

    sentences_list = parsed_dict["sentences"]["sentence"]

    for sentence in sentences_list:
        sentence["sentence_id"] = sentence["@id"]
        sentence["aspect_terms"] = []

        if "aspectTerms" in sentence:
            aspect_terms = sentence["aspectTerms"]["aspectTerm"]
            if isinstance(aspect_terms, dict):
                aspect_terms = [aspect_terms]
            for aspect_term in aspect_terms:
                aterm = {
                    "to": aspect_term["@to"],
                    "from": aspect_term["@from"],
                    "term": aspect_term["@term"],
                    "polarity": aspect_term["@polarity"],
                }
                sentence["aspect_terms"].append(aterm)

    df = pd.DataFrame(sentences_list)[["sentence_id", "text", "aspect_terms"]]

    return df


# %%
def load_reviews(path: Path, split: Optional[str] = None) -> pd.DataFrame:
    """
    Loads the reviews.
    - split: Data split to use. If None, all splits are used. Must be one of
        "train", "dev", "test".
    - path: Path to the data file in CSV format.
    """

    assert path.exists(), "Data file not found."

    aspect_terms = pd.read_csv(path, index_col="id")
    aspect_terms = aspect_terms[aspect_terms["split"] == split]
    reviews = aspect_terms.loc[:, ["text"]].drop_duplicates()

    return reviews


# %%
def load_system_messages(prompts_dir: Path, files: list) -> Dict[str, str]:
    """Load system messages from files in a directory."""
    system_message_dir = prompts_dir / "system_message"
    assert system_message_dir.exists(), "System message directory not found."

    system_message_files = [
        system_message_dir.joinpath(Path(f"{file}.txt")) for file in files
    ]

    def load_prompt(file: Path) -> str:
        assert file.exists(), f"File {file} not found."
        with open(file, "r") as f:
            return f.read()

    system_messages = {
        name: load_prompt(file) for name, file in zip(files, system_message_files)
    }

    return system_messages


# %%
def load_responses_from_disk(responses_path: Path):
    """Load model responses from a JSONL file."""
    assert responses_path.exists(), "Responses file not found."

    responses = pd.read_json(responses_path, lines=True).set_index(["id", "run_id"])
    responses["examples_in_separate_messages"] = responses[
        "examples_in_separate_messages"
    ].astype("boolean")

    # The examples_in_separate_messages parameters was added later.
    # Missing values mean True.
    responses["examples_in_separate_messages"].fillna(True, inplace=True)

    return responses


# %%
def upload_responses_to_s3(responses_path: Path, s3_path: str, s3_bucket: str):
    """Zip responses JSONL and upload to S3."""
    assert responses_path.exists(), "Responses file not found."

    temp = NamedTemporaryFile(suffix=".jsonl.gz")

    with gzip.open(temp.name, "wb") as f_out:
        with open(responses_path, "rb") as f_in:
            shutil.copyfileobj(f_in, f_out)

    s3 = boto3.client("s3")
    s3.upload_file(
        Filename=temp.name,
        Bucket=s3_bucket,
        Key=s3_path,
    )


# %%
def download_responses_from_s3(s3_path: str, responses_path: Path, s3_bucket: str):
    """Download responses from S3 and unzip."""

    temp = NamedTemporaryFile(suffix=".jsonl.gz")

    with open(temp.name, "wb") as f:
        s3 = boto3.client("s3")
        s3.download_file(
            Bucket=s3_bucket,
            Key=s3_path,
            Filename=f.name,
        )

    # Unzip the responses
    with gzip.open(temp.name, "rb") as f_in:
        with open(responses_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


# %%
def build_examples(example_ids: list[str], aspect_terms: pd.DataFrame) -> list[Example]:
    """Look up ids from the dataset and turn them to examples."""

    examples_df = aspect_terms.loc[example_ids][
        ["text", "term", "polarity", "split"]
    ].copy()

    assert all(
        examples_df["split"] == "train"
    ), "All examples must be in the training set."

    examples = []
    for id in examples_df.index.unique():
        text = examples_df.loc[[id], "text"].iloc[0]

        # Format in the same JSON format as specified in function to be called
        aspects = examples_df.loc[[id], ["term", "polarity"]]

        # Presence of NA means that the sentence doesn't contain any aspect terms
        if aspects["term"].isna().any():
            answer_json = json.dumps([])
        else:
            answer_json = json.dumps(aspects.to_dict(orient="records"))

        example = Example(context=text, completion=answer_json)
        examples.append(example)

    assert len(examples) == len(example_ids)

    return examples
