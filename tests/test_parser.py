import pytest
from absa_llm import parser, config

# Define test cases


# Test JSON parsing
def test_json_parsing():
    valid_json = '{"aspects": [{"term": "example", "polarity": "positive"}]}'
    assert parser.parse_answer_string(
        valid_json, config["paths"]["function_schema_path"]
    ) == [{"term": "example", "polarity": "positive"}]

    invalid_json = "not_json"
    assert (
        parser.parse_answer_string(
            invalid_json, config["paths"]["function_schema_path"]
        )
        is None
    )


# Test handling of different data structures
def test_data_structure_handling():
    # Test empty dictionary
    empty_dict_json = "{}"
    assert (
        parser.parse_answer_string(
            empty_dict_json, config["paths"]["function_schema_path"]
        )
        == []
    )

    # Test aspects as a list of dictionaries
    valid_aspects_list = '{"aspects": [{"term": "example1", "polarity": "positive"}, {"term": "example2", "polarity": "negative"}]}'
    assert parser.parse_answer_string(
        valid_aspects_list, config["paths"]["function_schema_path"]
    ) == [
        {"term": "example1", "polarity": "positive"},
        {"term": "example2", "polarity": "negative"},
    ]

    # Test aspects as a non-list dictionary
    invalid_aspects_dict = '{"aspects": {"term": "example", "polarity": "positive"}}'
    assert (
        parser.parse_answer_string(
            invalid_aspects_dict, config["paths"]["function_schema_path"]
        )
        is None
    )

    # Test malformed element in aspects list
    malformed_aspects_list = '{"aspects": [{"term": "example1", "polarity": "positive"}, {"invalid_key": "value"}]}'
    assert (
        parser.parse_answer_string(
            malformed_aspects_list, config["paths"]["function_schema_path"]
        )
        is None
    )

    # Test that non-schema generated lists of dictionaries get parsed properly
    aspects_list = '[{"term": "example", "polarity": "positive"}]'
    assert parser.parse_answer_string(
        aspects_list, config["paths"]["function_schema_path"]
    ) == [{"term": "example", "polarity": "positive"}]
