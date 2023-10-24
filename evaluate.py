# %%
# Evaluation script for the joint aspect extraction and sentiment classification task
# Writes Parquet files to the pinboard for visualization in the Quarto notebook

import logging
import json

import pandas as pd
import numpy as np

from pyprojroot import here

from absa_llm import config, loaders, board, parser
from absa_llm.job import count_tokens
from absa_llm.models import MODELS

results_dir = here(config["paths"]["results_dir"])

if not results_dir.exists():
    results_dir.mkdir()

logging.basicConfig(level=logging.INFO)
logging.getLogger("parser").setLevel(logging.CRITICAL)
logger = logging.getLogger("parser")

# %%
# Read the responses and expected aspects
responses_path = here(config["paths"]["responses_path"])
data_path = here(config["paths"]["aspect_terms_path"])

assert data_path.exists(), "Dataset file not found."

target_aspects_all = pd.read_csv(data_path, index_col="id")

if config["paths"]["use_s3"]:
    loaders.download_responses_from_s3(
        s3_path=config["paths"]["responses_s3_path"],
        responses_path=responses_path,
        s3_bucket=config["paths"]["s3_bucket"],
    )
responses = loaders.load_responses_from_disk(responses_path)

responses.rename({"n_train_examples": "in_context_examples"}, axis=1, inplace=True)

target_aspects = target_aspects_all.loc[responses.index.unique("id")]
assert set(responses.index.unique("id")) == set(target_aspects.index)

n_runs = len(responses.index.unique("run_id"))

print(f"Evaluating {n_runs} runs.")

# %%
# Give prompts short names for visualization
prompt_name_shortnames = {
    "semeval2014_guidelines_summary_gpt4": "Guidelines summary",
    "semeval2014_annotation_guidelines": "Annotation guidelines",
    "roleplay": "Roleplay",
    "semeval2014_reference": "Reference",
    "instructabsa_with_examples": "InstructABSA with examples",
    "separate_tasks": "Separate tasks",
    "instructabsa": "InstructABSA",
    "finetune": "Minimal",
    "empty": "Empty",
}

responses["system_message_shortname"] = responses["system_message_name"].map(
    prompt_name_shortnames
)

# Give models short names for visualization
model_name_shortnames = {
    "gpt-3.5-turbo-0613": "GPT-3.5",
    "gpt-4-0613": "GPT-4",
    "ft:gpt-3.5-turbo-0613:q-agentur-f-r-forschung:absa-finetune:81uJDAWD": "GPT-3.5 finetuned, minimal prompt",
    "ft:gpt-3.5-turbo-0613:q-agentur-f-r-forschung:absa-finetune:82j3P3BC": "GPT-3.5 finetuned, guidelines summary prompt",
    "ft:gpt-3.5-turbo-0613:q-agentur-f-r-forschung:absa-finetune:833uQmd6": "GPT-3.5 finetuned, no prompt",
}

responses["model_shortname"] = responses["model"].map(model_name_shortnames)

# %%
# Save number of train and test examples
example_count_by_split = (
    target_aspects_all.reset_index()
    .drop_duplicates(subset="id")
    .groupby("split")["id"]
    .count()
    .drop("dev", errors="ignore")
    .to_frame()
    .rename({"id": "examples"}, axis=1)
)

board.pin_write(
    example_count_by_split,
    name="example_count_by_split",
    type="parquet",
)

# %%
# Add domain and split information to the responses
id_domain_splits = target_aspects.loc[
    ~target_aspects.index.duplicated(keep="first"), ["domain", "split"]
]
responses = responses.join(id_domain_splits, on="id", how="inner")

# %%
# Remove blacklisted runs
blacklist = [
    "superb-marmot",
    "amused-earwig",
    "huge-grub",
    "robust-lizard",
    "real-tapir",
    "joint-emu"
]


def filter_rows_by_index_blacklist(df, blacklist):
    df_reset = df.reset_index()
    filtered_df = df_reset[~df_reset["run_id"].isin(blacklist)]
    filtered_df.set_index(["id", "run_id"], inplace=True)

    return filtered_df


responses = filter_rows_by_index_blacklist(responses, blacklist)

# %%
# Extract nested fields from the responses
usage = pd.json_normalize(responses["usage"])
usage.index = responses.index
responses = pd.concat([responses, usage], axis=1)

# %%
# Add pricing information
pricing = (
    pd.DataFrame(MODELS)
    .T[["price_prompt_tokens_1k", "price_completion_tokens_1k"]]
    .rename({"index": "model"}, axis=1)
)

responses = responses.join(pricing, on="model", how="left")

responses = responses.assign(
    price_prompt=lambda x: x["price_prompt_tokens_1k"] * x["prompt_tokens"] / 1000,
    price_completion=lambda x: x["price_completion_tokens_1k"]
    * x["completion_tokens"]
    / 1000,
    price_total=lambda x: x["price_prompt"] + x["price_completion"],
)

# %%
# Add the length of the system message in tokens
system_messages_dict = loaders.load_system_messages(
    here(config["paths"]["prompts_dir"]),
    responses["system_message_name"].unique(),
)
system_messages_df = pd.DataFrame.from_dict(
    system_messages_dict, orient="index", columns=["system_message"]
)
system_messages_df.index.name = "system_message_name"

system_messages_df["system_message_tokens"] = system_messages_df[
    "system_message"
].apply(count_tokens)

# Remove the actual system message
responses = responses.drop("system_message", axis=1)

responses = responses.join(system_messages_df, on="system_message_name", how="left")


# %%
# Extract the function call from the responses
def extract_predictions_from_choices(choices) -> list[dict]:
    answer_string = parser.retrieve_answer_string(choices)
    predictions = parser.parse_answer_string(
        answer_string, here(config["paths"]["function_schema_path"])
    )

    return predictions


responses["prediction"] = responses["choices"].apply(extract_predictions_from_choices)

# Record number of None values as parsing errors
responses["parsing_errors"] = responses["prediction"].apply(lambda x: x is None)

# Get the number of parsing errors by run
parsing_errors_by_run = (
    responses.reset_index()
    .groupby(["run_id", "domain", "split", "model"])
    .agg({"parsing_errors": "sum"})
    .rename({"parsing_errors": "parsing_errors_by_run"}, axis=1)
)

# Check that there are no duplicate columns in responses
assert not responses.columns.duplicated().any()


# %%
def calculate_metrics(row):
    # Fetch the target and predicted aspects
    index = row.name  # name of the row is index
    target = target_aspects.loc[[index[0]], ["term", "polarity"]]
    if target["term"].isnull().any():
        target_set = set()
    else:
        target_set = set(target.itertuples(index=False, name=None))
        target_set = {(t[0].lower(), t[1]) for t in target_set}

    pred = row["prediction"]
    if pred is None:
        pred_set = set()
    else:
        pred_set = set([(p["term"].lower(), p["polarity"]) for p in pred])

    # Compare the target and predicted aspects to calculate the metrics
    tp = len(target_set.intersection(pred_set))
    fp = len(pred_set.difference(target_set))
    fn = len(target_set.difference(pred_set))

    # Handle the case where there are no aspects in the target or prediction
    # This means the model correctly predicted "no aspects"
    # InstructABSA handles it this way
    if len(target_set) == 0 and len(pred_set) == 0:
        tp += 1

    return pd.Series([tp, fp, fn])


responses[["tp", "fp", "fn"]] = responses.apply(calculate_metrics, axis=1)


# %%
def precision(tp, fp):
    precision = np.where((tp + fp) == 0, 0, tp / (tp + fp))
    return precision.astype(float)


def recall(tp, fn):
    recall = np.where((tp + fn) == 0, 0, tp / (tp + fn))
    return recall.astype(float)


def f1_score(tp, fp, fn):
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    f1 = np.where((prec + rec) == 0, 0, 2 * (prec * rec) / (prec + rec))
    return f1.astype(float)


# %%
metrics_macro_by_domain = (
    responses.reset_index()
    .assign(examples=1)  # Count the number of examples
    .groupby(
        [
            "run_id",
            "domain",
            "system_message_name",
            "system_message_shortname",
            "system_message_tokens",
            "split",
            "temperature",
            "model",
            "model_shortname",
            "in_context_examples",
            "examples_in_separate_messages",
        ]
    )
    .agg(
        {
            "tp": "sum",
            "fn": "sum",
            "fp": "sum",
            "examples": "count",
            "price_total": "sum",
            "parsing_errors": "sum",
        }
    )
    .assign(
        precision=lambda df: precision(df["tp"], df["fp"]),
        recall=lambda df: recall(df["tp"], df["fn"]),
        f1=lambda df: f1_score(df["tp"], df["fp"], df["fn"]),
        price_per_example=lambda df: df["price_total"] / df["examples"],
    )
)

board.pin_write(
    metrics_macro_by_domain,
    name="metrics_macro_by_domain",
    type="parquet",
)


# %%
# Find best setup across domains
metrics_macro_cross_domain = (
    metrics_macro_by_domain.reset_index()
    .groupby(
        [
            "run_id",
            "split",
            "system_message_name",
            "system_message_shortname",
            "system_message_tokens",
            "in_context_examples",
            "model",
            "model_shortname",
            "examples_in_separate_messages",
        ]
    )
    .agg(
        {
            "tp": "sum",
            "fn": "sum",
            "fp": "sum",
            "examples": "sum",
            "price_total": "first",
        }
    )
    .assign(
        precision=lambda df: precision(df["tp"], df["fp"]),
        recall=lambda df: recall(df["tp"], df["fn"]),
        f1=lambda df: f1_score(df["tp"], df["fp"], df["fn"]),
    )
    .sort_values(by="f1", ascending=False)
)

board.pin_write(
    metrics_macro_cross_domain,
    name="metrics_macro_cross_domain",
    type="parquet",
)

# %%
# Check which prompt system message performs best
test_set_size = 1572

prompt_comparison = (
    metrics_macro_cross_domain.query(
        "split == 'test'"
        "& model == 'gpt-3.5-turbo-0613'"
        "& examples == @test_set_size"
        "& (examples_in_separate_messages == False | in_context_examples == 0)"
        "& system_message_name in @prompt_name_shortnames.keys()"
    )
    .reset_index()
    .drop_duplicates(
        subset=[
            "system_message_name",
            "system_message_tokens",
            "in_context_examples",
            "examples_in_separate_messages",
        ]
    )
)

board.pin_write(
    prompt_comparison,
    name="prompt_comparison",
    type="parquet",
)

# %%
# Error analysis

# Select the best run for each model
best_run_id_by_model = (
    metrics_macro_cross_domain.query("split == 'test' & examples == @test_set_size")
    .reset_index()
    .set_index("run_id")
    .groupby("model")["f1"]
    .idxmax()
)

best_run_responses = responses.copy().reset_index()
best_run_responses = best_run_responses[
    best_run_responses["run_id"].isin(best_run_id_by_model)
]
best_run_responses = best_run_responses[
    ["id", "run_id", "model_shortname", "tp", "fp", "fn", "prediction"]
].set_index("id")

# Prepare text and gold answers for error analysis
text_df = target_aspects.groupby("id")["text"].first().reset_index()

target_aspects_nested = (
    target_aspects.groupby("id")
    .apply(lambda group: group[["term", "polarity"]].to_dict("records"))
    .reset_index(name="aspects")
    .merge(text_df, on="id", how="inner")
)

errors_df = target_aspects_nested.merge(best_run_responses, on="id", how="outer")[
    ["id", "run_id", "model_shortname", "aspects", "prediction", "text", "fp"]
].dropna()


# %%
def count_wrong_polarity(row: dict) -> int:
    """Count the number of aspect terms with wrong polarity."""
    aspects = {d["term"]: d["polarity"] for d in row["aspects"]}
    wrong_polarity_count = 0
    for prediction in row["prediction"]:
        term = prediction["term"]
        if term in aspects and prediction["polarity"] != aspects[term]:
            wrong_polarity_count += 1
    return wrong_polarity_count


# %%
def count_terms_not_in_text(row: dict) -> int:
    """Count the number of aspect terms not in text."""
    prediction = row["prediction"]
    text = row["text"]

    # Convert 'text' to lowercase for case-insensitive comparison
    text_lower = text.lower()

    terms_not_in_text_count = 0

    for prediction_item in prediction:
        term = prediction_item["term"]
        term_lower = term.lower()

        if term_lower not in text_lower:
            terms_not_in_text_count += 1

    return terms_not_in_text_count


# %%

def count_partial_term_overlap(row: dict) -> int:
    """Count the number of aspect terms with partial overlap with gold terms."""
    # There is a very minor chance of this slightly over-estimating the number
    # of fp_aspect_boundary_errors in reviews where the terms are correctly identified
    # but are sub-strings of each other: e.g. in "We went to the place for their famous drinks,
    # but my drink wasn't very good" the aspects "drinks" and "drink" are subs-strings of
    # each other, leading to 2 aspect boundary errors. These cases are rare, and 
    # we deal with them by performing the error analysis only on reviews that have 
    # FPs in them. The possible edge cases where the sub-string matching could still 
    # be an issue are cases with another type of FP, along with this. 
    # This only potentially affects the error sub-type counting, not the primary metrics.
    
    prediction = row["prediction"]
    aspects = row["aspects"]

    # Convert 'term' values in 'prediction' and 'aspects' to lowercase for
    # case-insensitive comparison
    prediction_terms = [
        item["term"].lower() if isinstance(item["term"], str) else ""
        for item in prediction
    ]
    aspects_terms = [
        item["term"].lower() if isinstance(item["term"], str) else ""
        for item in aspects
    ]

    partial_overlap_count = 0

    for prediction_term in prediction_terms:
        # Check for partial overlap with terms in 'aspects'
        for aspect_term in aspects_terms:
            if (
                prediction_term
                and aspect_term
                and (prediction_term in aspect_term or aspect_term in prediction_term)
                and prediction_term != aspect_term
            ):
                partial_overlap_count += 1

    return partial_overlap_count


# %%
# Apply error analysis functions to the errors_df (filter to only rows with FPs on them)
errors_df["fp_wrong_polarity_count"] = errors_df.query("fp >= 1").apply(count_wrong_polarity, axis=1)
errors_df["fp_terms_not_in_text_count"] = errors_df.query("fp >= 1").apply(count_terms_not_in_text, axis=1)
errors_df["fp_aspect_boundary_errors"] = errors_df.query("fp >= 1").apply(
    count_partial_term_overlap, axis=1
)
# drop FP column
errors_df = errors_df.drop('fp', axis=1)

# %%
error_analysis = (
    errors_df.reset_index()
    .merge(
        metrics_macro_cross_domain.reset_index()[["run_id", "tp", "fp", "fn"]],
        on="run_id",
        how="left",
    )
    .loc[
        :,
        [
            "model_shortname",
            "tp",
            "fn",
            "fp",
            "fp_wrong_polarity_count",
            "fp_aspect_boundary_errors",
            "fp_terms_not_in_text_count",
        ],
    ]
    .groupby(["model_shortname", "tp", "fp", "fn"])
    .sum()
    .reset_index()
)
# add the last FP subtype after merging:
error_analysis["fp_terms_in_text_count"] = error_analysis["fp"] - error_analysis["fp_wrong_polarity_count"] - error_analysis["fp_aspect_boundary_errors"] -error_analysis["fp_terms_not_in_text_count"]


board.pin_write(
    error_analysis,
    name="error_analysis",
    type="parquet",
)

# %%
# Appendix: In-context examples
example_ids = config["examples"]["example_ids"]
aspect_terms = pd.read_csv(here(config["paths"]["aspect_terms_path"]), index_col="id")
examples = loaders.build_examples(example_ids, aspect_terms)

examples_df = pd.DataFrame.from_records([example.to_dict() for example in examples])

board.pin_write(
    examples_df,
    name="examples",
    type="parquet",
)


# %%
# Appendix: Prompts

# Find distinct values of system_message and system_message_name
system_messages_df = (
    responses.reset_index()[["system_message", "system_message_shortname"]]
    .drop_duplicates()
    .query("system_message_shortname != 'Empty'")
    .sort_values("system_message_shortname")
)

board.pin_write(
    system_messages_df,
    name="system_messages",
    type="parquet",
)


# %%
# Appendix: JSON schema
with open(here(config["paths"]["function_schema_path"]), "r") as f:
    function = json.load(f)

board.pin_write(
    function,
    name="function_schema",
    type="json",
)


# %%
