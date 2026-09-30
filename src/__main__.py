import argparse
import sys
from typing import List, Optional

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
        results.append(decoder.decode(prompt))
    if not write_results(args.output, results):
        sys.exit(1)


if __name__ == "__main__":
    main()
