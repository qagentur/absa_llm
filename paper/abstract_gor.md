# Large language models for aspect-based sentiment analysis

## Relevance & Research Question

Large language models (LLMs) like GPT-4 offer unprecedented text processing capabilities. As general models, they can fulfill a wide range of roles, including those of more specialized models. We investigated how well GPT-3.5 and 4 perform for aspect-based sentiment analysis (ABSA). ABSA is used for providing insights into digitized texts, such as product reviews or forum discussions, and is therefore a key capability for market research and computational social sciences.

## Methods & Data

We assess performance of GPT-3.5 and 4 both quantitatively and qualitatively. We evaluate performance on the gold standard benchmark dataset SemEval2014, consisting of human annotated laptop and restaurant reviews. Model performance is measured on a joint aspect term extraction and polarity classification task. We vary the prompt and the number of examples used and investigate the cost-accuracy tradeoff. We manually classify the errors made by the model and characterize its strengths and weaknesses.

## Results

Given 10 examples, GPT-4 outperforms BERT-based models trained on the full dataset, but does not reach the state of the art performance achieved by trained T5 models. The choice of prompt is crucial for performance and adding more examples improves performance further, however driving up the number of input tokens and therefore cost in the process. We discuss solutions such as bundling multiple prediction tasks into one prompt.

GPT-4's errors are typically related to the idiosyncrasies of the benchmark dataset and extensive labeling rules. It struggles to pick up on the nuances of labeling rules, instead occasionally delivering more commonsense labels. While such errors hamper benchmark performance, they should not necessarily discourage from using LLMs in real-world applications of ABSA or similar tasks.

## Added Value

This study provides market researchers evidence on the capabilities of LLMs for ABSA. It also provides practical hints for prompt engineering when using LLMs for structured extraction and classification tasks. By extension, it also helps with placing LLMs in contrast with specialized models.
