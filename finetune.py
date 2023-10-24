# Fine tune GPT-3.5 turbo on the training data
# Upload training log to the pinboard for visualization in Quarto

# %%
import pandas as pd
from pyprojroot import here
import json
import openai
import random

from absa_llm import loaders, config, board

random.seed(42)

finetune_dir = here(config["paths"]["finetune_dir"])
results_dir = here(config["paths"]["results_dir"])

# %%
# Load the training data
aspect_terms = pd.read_csv(here(config["paths"]["aspect_terms_path"]), index_col="id")
train_data = aspect_terms[aspect_terms["split"] == "train"]

system_message_name = config["finetune"]["system_message_name"]

system_message = loaders.load_system_messages(
    prompts_dir=here(config["paths"]["prompts_dir"]), files=[system_message_name]
)[system_message_name]


# %%
# Create examples in chat format
def make_gold_answer(aspect_terms):
    if len(aspect_terms) == 0:
        return {}

    if len(aspect_terms) == 1:
        # Missing means no aspects found
        if pd.isna(aspect_terms["term"].iloc[0]):
            return {}

    return aspect_terms.to_dict(orient="records")


# %%
def built_finetune_examples(aspect_terms):
    """Build finetune examples from the aspect terms."""
    finetune_examples = []
    for id in aspect_terms.index:
        text = aspect_terms.loc[[id], "text"][0]
        aspects = aspect_terms.loc[[id], ["term", "polarity"]]
        gold_answer = json.dumps(make_gold_answer(aspects))

        example = {
            "messages": [
                {"role": "user", "content": text},
                {"role": "assistant", "content": gold_answer},
            ]
        }

        if system_message != "":
            example["messages"].insert(0, {"role": "system", "content": system_message})

        finetune_examples.append(example)

    return finetune_examples


# %%
# Create examples and shuffle them
finetune_examples = built_finetune_examples(train_data)
finetune_examples = random.sample(finetune_examples, len(finetune_examples))

# Randomly split into train and validation
train_size = int(len(finetune_examples) * 0.8)
train_examples = finetune_examples[:train_size]
validation_examples = finetune_examples[train_size:]

# %%
datasets = {
    "train": {
        "examples": train_examples,
        "path": finetune_dir / f"train_{system_message_name}.jsonl",
    },
    "validation": {
        "examples": validation_examples,
        "path": finetune_dir / f"validation_{system_message_name}.jsonl",
    },
}

# %%
# Upload the datasets to OpenAI
for name in datasets.keys():
    path = datasets[name]["path"]
    with open(path, "w") as f:
        for example in datasets[name]["examples"]:
            f.write(json.dumps(example) + "\n")

    response = openai.File.create(
        file=open(path, "rb"),
        purpose="fine-tune",
        user_provided_filename=path.name,
    )
    datasets[name]["response"] = response
    datasets[name]["file_id"] = response.id


# %%
# Run the fine-tuning job
openai.FineTuningJob.create(
    training_file=datasets["train"]["file_id"],
    validation_file=datasets["validation"]["file_id"],
    suffix=f"absa-finetune-{system_message_name}",
    model="gpt-3.5-turbo-0613",
)

# %%
# Get the results
job = openai.FineTuningJob.list()["data"][0]  # first element is the latest job
result_file_id = job["result_files"][0] # one results file
result_content = openai.File.download(result_file_id)

# %%
# Check the results
results_df = pd.read_csv(pd.io.common.BytesIO(result_content), index_col="step")

output_file_name = f"finetune_results_{system_message_name}"
board.pin_write(x=results_df, name=output_file_name, type="parquet")


# %%
