# %%
import json
import jsonschema
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger("parser")

# %%
# function that retrieves the answer string
def retrieve_answer_string(choices:list) -> str:
    if "function_call" in choices[0]["message"]:
        answer_string = choices[0]["message"]["function_call"]["arguments"]
    else:
        answer_string = choices[0]["message"]["content"]
    
    return answer_string
        
# %%
# function that parses the string into a list of dictionaries

def parse_answer_string(answer_string: str, schema_path: Path) -> List[Dict] | None:
    """
    Parses the function call from a hopefully JSON-formatted string
    into a list of dicts.
    Returns None if the string is not JSON-formatted.
    """

    try:
        answer_dict = json.loads(answer_string)
    except json.JSONDecodeError:
        logger.error("answer_string is not JSON-formatted")
        return None
    
    # Load the schema
    with open(schema_path, "r") as f:
        schema = json.load(f)
    
    try:
        jsonschema.validate(answer_dict, schema)
    except jsonschema.exceptions.ValidationError as e:
        # Dictionary is not valid against the schema
        logger.error("JSON schema validation error:", e.message)
        return None

    # Sometimes the arguments are a list of aspects, sometimes a dict
    if not isinstance(answer_dict, dict):
        # trying to format a list of dicts into a dict
        answer_dict = {'aspects': answer_dict}
        logger.error("Answer is not a dictionary. Formatting.")
    
    if answer_dict == {}:
        return []

    answer_array = answer_dict.get("aspects")
    
    if not isinstance(answer_array, list):
        logger.error("Aspects are not a list.")
        return None
    
    if len(answer_array) == 0:
        return answer_array
      
    if not all(isinstance(item, dict) and 'term' in item and 'polarity' in item for item in answer_array):
        logger.error("Answer array element malformed.")
        return None
    
    return answer_array
  
