# Run job.py with different parameters to see how the results change.
# All results are saved in a single .jsonl file that can be analyzed
# with the evaluate.py script.

# %%
import click
import itertools
import json
import pandas as pd

import absa_llm.job as job
import absa_llm.loaders as loaders
from absa_llm import config

from pyprojroot import here

# %%
# Load target terms
aspect_terms = pd.read_csv(here(config["paths"]["aspect_terms_path"]), index_col="id")

# Choose review ids to run inference on
data_split = config["split"]["split"]

assert data_split in ["dev", "test"], "Invalid data split to run."

# Choose ids of examples to run inference on
reviews = loaders.load_reviews(
    here(config["paths"]["aspect_terms_path"]), split=data_split
)

# Subsample the reviews to save money
max_examples = config["split"]["max_examples"]
reviews = reviews.sample(n=min(max_examples, len(reviews)), random_state=42)
print(f"Running inference on {len(reviews)} reviews.")

# %%
# Prepare examples
# Pick examples manually and save their ids in the config file
example_ids = config["examples"]["example_ids"]
example_count = config["parameters"]["example_count"]
assert len(example_ids) >= max(example_count), "Not enough examples in the config file"

# Build variants with different number of examples
example_id_list = [example_ids[:i] for i in example_count]

# %%
# Load system messages
system_messages = loaders.load_system_messages(
    prompts_dir=here(config["paths"]["prompts_dir"]),
    files=config["prompts"]["system_message_names"],
)

# %%
# Build parameter ranges
param_ranges = {
    "model": config["models"]["active_models"],
    "temperature": config["parameters"]["temperature"],
    "max_tokens": config["parameters"]["max_tokens"],
    "top_p": config["parameters"]["top_p"],
    "presence_penalty": config["parameters"]["presence_penalty"],
    "frequency_penalty": config["parameters"]["frequency_penalty"],
    "logit_bias_literal": config["parameters"]["logit_bias"]["words"],
    "example_ids": example_id_list,
    "system_message_name": list(system_messages.keys()),
    "examples_in_separate_messages": config["examples"][
        "examples_in_separate_messages"
    ],
    "use_function_calling": config["prompts"]["use_function_calling"],
}

# %%
# Build parameter grid from ranges
# Create all combinations of parameters
keys, values = zip(*param_ranges.items())
params_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

# Add the full system messages to the parameters
for p in params_list:
    p["system_message"] = system_messages[p["system_message_name"]]

# %%
# Turn the example ids into full examples
for p in params_list:
    p["train_examples"] = loaders.build_examples(p["example_ids"], aspect_terms)

    del p["example_ids"]

# %%
# Load function schema used for function-calling with ChatCompletion
with open(here(config["paths"]["function_schema_path"]), "r") as f:
    function = json.load(f)

# %%
# Create id-text pairs from reviews
id_text_pairs = [
    job.IdTextPair(id=id, text=text)
    for id, text in reviews.reset_index().itertuples(index=False, name=None)
]

# %%
runs = []
for p in params_list:
    params = job.Params(**p)

    run = job.ChatCompletionJob(
        responses_path=here(config["paths"]["responses_path"]),
        id_text_pairs=id_text_pairs,
        params=params,
        function=function if params.use_function_calling else None,
    )
    runs.append(run)

print(f"Created grid with {len(runs)} runs")


# %%
def run_inference(runs: list[job.ChatCompletionJob]):
    total_cost_max = sum([run.estimate_cost() for run in runs])

    if click.confirm(
        f"Do you want to execute the runs? Maximum cost is ${total_cost_max:.2f}",
        default=False,
    ):
        # Run inference
        for i, run in enumerate(runs):
            print(f"Run {i+1}/{len(runs)}")
            run.get_completions()
            run.save_completions()

            if config["paths"]["use_s3"]:
                print("Uploading responses to S3")
                loaders.upload_responses_to_s3(
                    responses_path=here(config["paths"]["responses_path"]),
                    s3_path=config["paths"]["responses_s3_path"],
                    s3_bucket=config["paths"]["s3_bucket"],
                )
    else:
        print("Aborted.")


run_inference(runs)
