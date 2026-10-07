import json
from pathlib import Path
from typing import Any, List, Optional

from pydantic import ValidationError

from src.models import FunctionDefinition, PromptInput, FunctionCall


def read_json_file(file_name: str) -> Optional[Any]:
    """Read the JSON file file_name and return its data, or None."""
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: file not found: {file_name}")
    except OSError as e:
        print(f"Error: cannot read {file_name}: {e}")
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"Error: {file_name} is not valid JSON: {e}")
    return None


def load_functions(file_name: str) -> Optional[List[FunctionDefinition]]:
    """Load the functions from file_name and return them, or None."""
    data = read_json_file(file_name)
    if data is None:
        return None
    if not isinstance(data, list):
        print(f"Error: {file_name} must contain a JSON list")
        return None

    functions: List[FunctionDefinition] = []
    try:
        for item in data:
            functions.append(FunctionDefinition.model_validate(item))
    except ValidationError as e:
        print(f"Error: invalid function in {file_name}:\n{e}")
        return None

    names = [function.name for function in functions]
    for name in names:
        if names.count(name) > 1:
            print(f"Error: function {name} is defined twice in {file_name}")
            return None
    return functions


def load_prompts(file_name: str) -> Optional[List[PromptInput]]:
    """Load the prompts from file_name and return them, or None."""
    data = read_json_file(file_name)
    if data is None:
        return None
    if not isinstance(data, list):
        print(f"Error: {file_name} must contain a JSON list")
        return None

    prompts: List[PromptInput] = []
    try:
        for item in data:
            prompts.append(PromptInput.model_validate(item))
    except ValidationError as e:
        print(f"Error: invalid prompt in {file_name}:\n{e}")
        return None
    return prompts


def write_results(file_name: str, results: List[FunctionCall]) -> bool:
    """Write the results to file_name as JSON and return True on success."""
    data = [result.model_dump() for result in results]

    try:
        path = Path(file_name)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    except OSError as e:
        print(f"Error: cannot write {file_name}: {e}")
        return False
    return True
