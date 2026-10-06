from typing import List

from src.models import FunctionDefinition


def format_function(function: FunctionDefinition) -> str:
    """Return a one-line description of function for the LLM."""
    params = ", ".join(
        f"{name}: {param.type}"
        for name, param in function.parameters.items()
    )

    return f"- {function.name}({params}): {function.description}"


def build_prompt(prompt: str, functions: List[FunctionDefinition]) -> str:
    """Return the text given to the LLM for prompt and functions."""
    lines = ["Available functions:"]

    for function in functions:
        lines.append(format_function(function))

    lines.append("")
    lines.append("Copy values exactly as written in the question.")
    lines.append(f"Question: {prompt}")
    lines.append('Answer: {"name": "')

    return "\n".join(lines)
