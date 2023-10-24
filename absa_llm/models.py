# Information about OpenAI models
# See https://platform.openai.com/docs/models
# and https://platform.openai.com/docs/guides/rate-limits/overview

MODELS = {
    "gpt-3.5-turbo-0613": {
        "context_tokens": 4096,
        "price_prompt_tokens_1k": 0.0015,
        "price_completion_tokens_1k": 0.002,
        "max_requests_per_minute": 3500,
        "max_tokens_per_minute": 90000,
    },
    "gpt-4-0613": {
        "context_tokens": 8192,
        "price_prompt_tokens_1k": 0.03,
        "price_completion_tokens_1k": 0.06,
        "max_requests_per_minute": 200,
        "max_tokens_per_minute": 40000,
    },
    "ft:gpt-3.5-turbo-0613:q-agentur-f-r-forschung:absa-finetune:81uJDAWD": {
        "context_tokens": 4096,
        "price_prompt_tokens_1k": 0.012,
        "price_completion_tokens_1k": 0.016,
        "max_requests_per_minute": 3500,
        "max_tokens_per_minute": 90000,
    },
    "ft:gpt-3.5-turbo-0613:q-agentur-f-r-forschung:absa-finetune:82j3P3BC": {
        "context_tokens": 4096,
        "price_prompt_tokens_1k": 0.012,
        "price_completion_tokens_1k": 0.016,
        "max_requests_per_minute": 3500,
        "max_tokens_per_minute": 90000,
    },
    "ft:gpt-3.5-turbo-0613:q-agentur-f-r-forschung:absa-finetune:833uQmd6": {
        "context_tokens": 4096,
        "price_prompt_tokens_1k": 0.012,
        "price_completion_tokens_1k": 0.016,
        "max_requests_per_minute": 3500,
        "max_tokens_per_minute": 90000,
    }
}
