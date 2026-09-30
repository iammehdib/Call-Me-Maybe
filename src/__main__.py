import argparse
import sys
from typing import Dict, List, Optional, Union

from llm_sdk import Small_LLM_Model

from src.decoder import Decoder
from src.models import FunctionCall, FunctionDefinition, PromptInput
from src.utils import load_functions, load_prompts, write_results


def main() -> None:
    """Parse the command line arguments and run the program."""
    parser = argparse.ArgumentParser(
        prog="Call-Me-Maybe",
        description="Translate prompts into function calls."
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calls.json",
    )

    args = parser.parse_args()

    functions: Optional[List[FunctionDefinition]] = (
        load_functions(args.functions_definition))
    prompts: Optional[List[PromptInput]] = (
        load_prompts(args.input))
    if prompts is None or functions is None:
        sys.exit(1)
    if not functions:
        print("Error: empty functions defined in file")
        sys.exit(1)

    model: Small_LLM_Model = Small_LLM_Model()
    decoder: Decoder = Decoder(model, functions)
    results: List[FunctionCall] = []
    for prompt in prompts:
        lines: str = decoder.build_prompt(prompt.prompt)
        print(lines)
        results.append(fake_call(prompt, functions[0]))
    if not write_results(args.output, results):
        sys.exit(1)


def fake_call(
    prompt: PromptInput, function: FunctionDefinition
) -> FunctionCall:
    """Return a placeholder call to function for prompt."""
    parameters: Dict[str, Union[float, int, str, bool]] = {}
    for name, param_type in function.parameters.items():
        match param_type.type:
            case "number":
                parameters[name] = 0.0
            case "integer":
                parameters[name] = 0
            case "string":
                parameters[name] = ""
            case "boolean":
                parameters[name] = False

    return FunctionCall(
        prompt=prompt.prompt,
        name=function.name,
        parameters=parameters
    )


if __name__ == "__main__":
    main()
