import argparse
import sys
from typing import List, Optional

from llm_sdk import Small_LLM_Model

from src.decoder import Decoder
from src.models import FunctionCall, FunctionDefinition, PromptInput
from src.files import load_functions, load_prompts, write_results


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
        print(f"Error: no functions defined in {args.functions_definition}")
        sys.exit(1)

    model: Small_LLM_Model = Small_LLM_Model()
    decoder: Decoder = Decoder(model, functions)
    results: List[FunctionCall] = []
    for index, prompt in enumerate(prompts, start=1):
        result = decoder.decode(prompt)
        print(f"[{index}/{len(prompts)}] {result.name} <- {prompt.prompt}")
        results.append(result)

    if not write_results(args.output, results):
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by the user.")
        sys.exit(1)
    except Exception as error:
        print(f"Error: {error}")
        sys.exit(1)
