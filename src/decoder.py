import json
from typing import ClassVar, Dict, List, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr

from llm_sdk import Small_LLM_Model
from src.models import FunctionCall, FunctionDefinition, PromptInput
from src.utils import is_start

MAX_TOKENS = 40


class Decoder(BaseModel):
    """Turn prompts into function calls with constrained decoding."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vocab_cache: ClassVar[Dict[str, List[str]]] = {}

    model: Small_LLM_Model
    functions: List[FunctionDefinition]

    _vocab: List[str] = PrivateAttr(default_factory=list)
    _digit_ids: List[int] = PrivateAttr(default_factory=list)
    _end_ids: List[int] = PrivateAttr(default_factory=list)
    _text_ids: List[int] = PrivateAttr(default_factory=list)
    _minus_id: int = PrivateAttr(default=0)
    _dot_id: int = PrivateAttr(default=0)
    _quote_id: int = PrivateAttr(default=0)
    _ids: List[int] = PrivateAttr(default_factory=list)

    def __init__(
        self, model: Small_LLM_Model, functions: List[FunctionDefinition]
    ) -> None:
        """Validate the fields, then get the vocabulary of model."""
        super().__init__(model=model, functions=functions)
        self._vocab = self.get_vocab(model)

        self._digit_ids = [self._vocab.index(digit) for digit in "0123456789"]
        self._end_ids = [self._vocab.index(","), self._vocab.index("}")]
        self._minus_id = self._vocab.index("-")
        self._dot_id = self._vocab.index(".")
        self._quote_id = self._vocab.index('"')

        for token_id, token in enumerate(self._vocab):
            if token != "" and '"' not in token and "\n" not in token:
                self._text_ids.append(token_id)

    @classmethod
    def get_vocab(cls, model: Small_LLM_Model) -> List[str]:
        """Return the text of every token of model, cached per vocab file."""
        path = model.get_path_to_vocab_file()
        if path in cls.vocab_cache:
            return cls.vocab_cache[path]

        with open(path, "r", encoding="utf-8") as f:
            token_to_id: Dict[str, int] = json.load(f)

        vocab: List[str] = []
        for token_id in range(len(token_to_id)):
            vocab.append(model.decode([token_id]))
        cls.vocab_cache[path] = vocab
        return vocab

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
        return self._vocab[token_id]

    def pick_token(self, allowed_ids: List[int]) -> int:
        """Return the allowed token the LLM prefers to write next."""
        logits = np.array(self.model.get_logits_from_input_ids(self._ids))

        masked = np.full_like(logits, -np.inf)
        masked[allowed_ids] = logits[allowed_ids]

        return int(np.argmax(masked))

    def word_choices(self, result: str, words: List[str]) -> List[int]:
        """Return the tokens that keep result on the way to one of words."""
        choices: List[int] = []
        for token_id, token in enumerate(self._vocab):
            if token != "" and is_start(result + token, words):
                choices.append(token_id)
        return choices

    def pick_word(self, words: List[str]) -> str:
        """Make the LLM write one of words and return it."""
        result = ""

        for _ in range(MAX_TOKENS):
            if result in words:
                break

            chosen = self.pick_token(self.word_choices(result, words))
            result += self.add_token(chosen)

        return result

    def number_choices(self, result: str, integer: bool) -> List[int]:
        """Return the tokens allowed after result in a number."""
        is_empty = result == ""
        has_digit = any(char.isdigit() for char in result)
        has_dot = "." in result
        ends_with_dot = result.endswith(".")

        choices = list(self._digit_ids)

        if is_empty:
            choices.append(self._minus_id)

        if not integer and has_digit and not has_dot:
            choices.append(self._dot_id)

        if has_digit and not ends_with_dot:
            choices += self._end_ids

        return choices

    def pick_number(self, integer: bool) -> str:
        """Make the LLM write a number and return it."""
        result = ""

        for _ in range(MAX_TOKENS):
            chosen = self.pick_token(self.number_choices(result, integer))
            if chosen in self._end_ids:
                break

            result += self.add_token(chosen)

        return result

    def pick_string(self) -> str:
        """Make the LLM write a string and return it."""
        allowed_ids = self._text_ids + [self._quote_id]
        result = ""

        for _ in range(MAX_TOKENS):
            chosen = self.pick_token(allowed_ids)
            if chosen == self._quote_id:
                break

            result += self.add_token(chosen)

        return result

    def _pick_parameters(
        self, function: FunctionDefinition
    ) -> Dict[str, Union[float, int, str, bool]]:
        """Return the value the LLM writes for every parameter."""
        params: Dict[str, Union[float, int, str, bool]] = {}
        self.add_text('", "parameters": {')

        for key, param in function.parameters.items():
            if params:
                self.add_text(", ")
            self.add_text(f'"{key}": ')

            match param.type:
                case "number":
                    params[key] = float(self.pick_number(integer=False))
                case "integer":
                    params[key] = int(self.pick_number(integer=True))
                case "string":
                    self.add_text('"')
                    params[key] = self.pick_string()
                    self.add_text('"')
                case "boolean":
                    params[key] = self.pick_word(["true", "false"]) == "true"

        return params

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
