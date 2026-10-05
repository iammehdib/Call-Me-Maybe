from typing import Dict, List, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr

from llm_sdk import Small_LLM_Model
from src.models import FunctionCall, FunctionDefinition, PromptInput
from src.utils import is_start, unescape
from src.vocabulary import Vocabulary

MAX_TOKENS = 40


class Decoder(BaseModel):
    """Turn prompts into function calls with constrained decoding."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Small_LLM_Model
    functions: List[FunctionDefinition]

    _vocab: Vocabulary = PrivateAttr()
    _ids: List[int] = PrivateAttr(default_factory=list)

    def __init__(
        self, model: Small_LLM_Model, functions: List[FunctionDefinition]
    ) -> None:
        """Validate the fields, then get the vocabulary of model."""
        super().__init__(model=model, functions=functions)
        self._vocab = Vocabulary(model)

    def decode(self, prompt_input: PromptInput) -> FunctionCall:
        """Return the function call the LLM chooses for prompt_input."""
        self._ids = []
        self.add_text(self.build_prompt(prompt_input.prompt))

        names = [function.name for function in self.functions]
        name = self.pick_word(names)
        function = self.functions[names.index(name)]

        return FunctionCall(
            prompt=prompt_input.prompt,
            name=name,
            parameters=self._pick_parameters(function),
        )

    @staticmethod
    def format_function(function: FunctionDefinition) -> str:
        """Return a one-line description of function for the LLM."""
        params: str = ", ".join(
            f"{name}: {param.type}"
            for name, param in function.parameters.items()
        )

        return f"- {function.name}({params}): {function.description}"

    def build_prompt(self, prompt: str) -> str:
        """Return the text given to the LLM for prompt."""
        lines: List[str] = ["Available functions:"]

        for function in self.functions:
            lines.append(self.format_function(function))

        lines.append("")
        lines.append("Copy values exactly as written in the question.")
        lines.append(f"Question: {prompt}")
        lines.append('Answer: {"name": "')

        return "\n".join(lines)

    def add_text(self, text: str) -> None:
        """Add the tokens of a fixed text to what the LLM reads."""
        text_ids: List[int] = self.model.encode(text)[0].tolist()
        self._ids = self._ids + text_ids

    def add_token(self, token_id: int) -> str:
        """Add one token to what the LLM reads and return its text."""
        self._ids = self._ids + [token_id]
        return self._vocab.texts[token_id]

    def pick_token(self, allowed_ids: List[int]) -> int:
        """Return the allowed token the LLM prefers to write next."""
        logits = np.array(self.model.get_logits_from_input_ids(self._ids))

        masked = np.full_like(logits, -np.inf)
        masked[allowed_ids] = logits[allowed_ids]

        return int(np.argmax(masked))

    def word_choices(self, result: str, words: List[str]) -> List[int]:
        """Return the tokens that keep result on the way to one of words."""
        choices: List[int] = []

        for token_id, token in enumerate(self._vocab.texts):
            if token != "" and is_start(result + token, words):
                choices.append(token_id)

        return choices

    def pick_word(self, words: List[str]) -> str:
        """Make the LLM write one of words and return it."""
        result = ""

        for _ in range(MAX_TOKENS):
            if result in words:
                break

            # Only one word left: write the rest without asking the LLM.
            remaining = [word for word in words if word.startswith(result)]
            if len(remaining) == 1:
                self.add_text(remaining[0][len(result):])
                return remaining[0]

            chosen = self.pick_token(self.word_choices(result, words))
            result += self.add_token(chosen)

        return result

    def number_choices(self, result: str, integer: bool) -> List[int]:
        """Return the tokens allowed after result in a number."""
        # After '"a":' the LLM writes " 42" or " -42": start with " " or " -".
        if result == "":
            return [self._vocab.space_id, self._vocab.space_minus_id]

        has_digit = any(char.isdigit() for char in result)
        choices = list(self._vocab.digit_ids)

        has_dot = "." in result
        if not integer and has_digit and not has_dot:
            choices.append(self._vocab.dot_id)

        if has_digit and not result.endswith("."):
            choices += self._vocab.end_ids

        return choices

    def pick_number(self, integer: bool) -> str:
        """Make the LLM write a number and return it."""
        result = ""

        for _ in range(MAX_TOKENS):
            chosen = self.pick_token(self.number_choices(result, integer))
            if chosen in self._vocab.end_ids:
                break

            result += self.add_token(chosen)

        return result

    def pick_string(self) -> str:
        """Make the LLM write a string and return it."""
        allowed_ids = self._vocab.text_ids + self._vocab.quote_ids
        result = ""

        for _ in range(MAX_TOKENS):

            chosen = self.pick_token(allowed_ids)

            if chosen in self._vocab.quote_ids:
                # A token like '",' ends the string: keep what is before '"'.
                before_quote = self._vocab.texts[chosen].split('"')[0]
                if (result + before_quote).endswith("\\"):
                    # '\"' is a quote inside the string: keep writing.
                    result += self.add_token(chosen)
                    continue
                if before_quote.strip() != "":
                    self.add_text(before_quote)
                    result += before_quote
                break

            result += self.add_token(chosen)

        return unescape(result).lstrip()

    def _pick_parameters(
        self, function: FunctionDefinition
    ) -> Dict[str, Union[float, int, str, bool]]:
        """Return the value the LLM writes for every parameter."""
        params: Dict[str, Union[float, int, str, bool]] = {}
        self.add_text('", "parameters": {')

        for key, param in function.parameters.items():
            if params:
                self.add_text(", ")
            self.add_text(f'"{key}":')

            match param.type:
                case "number":
                    params[key] = float(self.pick_number(integer=False))
                case "integer":
                    params[key] = int(self.pick_number(integer=True))
                case "string":
                    self.add_text(" ")
                    self.add_text('"')
                    params[key] = self.pick_string()
                    self.add_text('"')
                case "boolean":
                    self.add_text(" ")
                    params[key] = self.pick_word(["true", "false"]) == "true"

        return params
