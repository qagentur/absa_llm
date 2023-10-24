# %%
import numpy as np
import pandas as pd

from absa_llm.loaders import read_semeval_xml
from absa_llm import config
from pyprojroot import here


# %%
# Load XML files
# Train and dev data sets downloaded from:
# https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools#
# Test data set with gold standard labels downloaded from:
# http://metashare.ilsp.gr:8080/repository/download/b98d11cec18211e38229842b2b6a04d77591d40acd7542b7af823a54fb03a155/
data_dir = here(config["paths"]["data_dir"] + "semeval2014/")

splits = ["train", "dev", "test"]
domains = ["laptops", "restaurants"]

reviews = pd.DataFrame()

for split in splits:
    for domain in domains:
        filename = f"{domain}_{split}.xml"
        filepath = data_dir / filename
        assert filepath.exists(), f"File not found: {filepath}"
        df = read_semeval_xml(filepath)
        df["split"] = split
        df["domain"] = domain
        reviews = pd.concat([reviews, df])

assert set(reviews.columns) == {
    "sentence_id",
    "text",
    "aspect_terms",
    "split",
    "domain",
}


# %%
# Make a unique index id. By default, sentence_id is not unique because
# it is reused for each domain
reviews["id"] = reviews["sentence_id"].astype(str) + "_" + reviews["domain"].astype(str)

# %%
# Drop a mistaken from train because it's also in the test set
reviews = reviews[~((reviews["id"] == "227_laptops") & (reviews["split"] == "train"))]

# %%
# By default, the dev set has ids that are also in the training set
# This is not the case for the test set. We remove the duplicates
# from the train set, keeping the occurrence in the dev set

mask = (reviews["split"] == "train") & reviews.duplicated(subset="id", keep="last")

assert mask.sum() == 200  # dev set size

# Delete the masked rows
reviews = reviews[~mask].copy()
reviews.set_index("id", inplace=True)

assert reviews.index.is_unique


# %%
# Turn into long format
exploded = reviews["aspect_terms"].explode()
aspect_terms = pd.json_normalize(exploded)
aspect_terms.set_index(exploded.index, inplace=True)

# %%
# The original challenge included a "conflict" polarity value
# This was not used in further studies and we don't use it either
# Drop complete examples with conflict polarity
conflict_example_ids = aspect_terms.index[
    aspect_terms["polarity"] == "conflict"
].unique()

conflict_split_counts = reviews.loc[conflict_example_ids].split.value_counts()

# Drop the conflict examples from the reviews
reviews.drop(conflict_example_ids, inplace=True)
aspect_terms.drop(conflict_example_ids, inplace=True)

assert reviews["split"].value_counts()["train"] == 5759  # dropped 126 conflict examples
assert reviews["split"].value_counts()["test"] == 1572  # dropped 28 conflict examples
assert reviews["split"].value_counts()["dev"] == 200  # dropped no conflict examples

# np.nan means that there is no aspect term in the sentence
assert aspect_terms["polarity"].isin(["positive", "negative", "neutral", np.nan]).all()

# %%
# Merge the aspect terms with the original reviews
cleaned = reviews.merge(aspect_terms, how="left", left_index=True, right_index=True)
cleaned = cleaned[["text", "from", "to", "term", "polarity", "domain", "split"]]

# %%
# Save the reviews
cleaned.to_csv(here(config["paths"]["aspect_terms_path"]))

# %%
