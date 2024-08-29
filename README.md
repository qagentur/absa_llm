# Aspect based sentiment analysis using LLMs

This repository contains the code for the paper "Large language models for aspect-based sentiment analysis" by Paul Simmering and Paavo Huoviala.

## Task and benchmark

The task is Aspect-Based Sentiment Analysis (ABSA) on SemEval 2014 Task 4 Subtask 1+2. See the benchmark on [Papers with Code](https://paperswithcode.com/sota/aspect-based-sentiment-analysis-on-semeval-6).

We are unable to share the data-set used directly, but it is available from:
 http://metashare.elda.org/repository/browse/semeval-2014-absa-test-data-gold-annotations/b98d11cec18211e38229842b2b6a04d77591d40acd7542b7af823a54fb03a155/

Organize the files like this:

```
data/semeval2024/laptops_dev.xml
data/semeval2024/laptops_test.xml
data/semeval2024/laptops_train.xml
data/semeval2024/restaurants_dev.xml
data/semeval2024/restaurants_test.xml
data/semeval2024/restaurants_train.xml
```

Given a review sentence, the task is to extract the aspects and their polarity.

Example input:

> The food was great, but the service was awful.

Example output:

```{json}
{
    "aspects":
        [
            {"term": "food", "polarity": "positive"},
            {"term": "service", "polarity": "negative"}
        ]
}
```

We test if OpenAI's GPT-models can solve this task in a zero-shot, few-shot setting, and fine-tuned settings.

## Requirements

- Python 3.10.9
- Packages listed in pyproject.toml, install with `poetry install`
- OpenAI API key placed in a file called `.env` in the root directory of the repository

Please note that using the OpenAI API costs money.

The [pins package](https://pypi.org/project/pins/) is used for storing the model outputs and the evaluation results. By default the S3 backend of AWS is used. 

Please create a `.env` file in the root directory of the repository with the following contents:

```{bash}
OPENAI_API_KEY=<your key>
AWS_ACCESS_KEY_ID=<your key>
AWS_SECRET_ACCESS_KEY`=<your key>
```

## Using this project

1. Download and save the XML files as detailed above
2. Run `python load_data.py` and verify that `data/cleaned.csv` was created
3. Adjust settings in `project.cfg` and prompts in the `prompts/` directory.
4. Run `python finetune.py` to finetune a model, if required.
5. Run `python hyper.py` to get the results. Check the cost and approve the run.
6. Run `python evaluate.py` to parse and evaluate the model outputs.
7. Run the Quarto notebook `paper/paper.qmd` to visualize the results and typeset the paper.
