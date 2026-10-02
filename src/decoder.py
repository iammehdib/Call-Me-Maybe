import json
from typing import Callable, ClassVar, Dict, List, Tuple, Union

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

    def __init__(
        self, model: Small_LLM_Model, functions: List[FunctionDefinition]
    ) -> None:
        """Validate the fields, then get the vocabulary of model."""
        super().__init__(model=model, functions=functions)
        self._vocab = self.get_vocab(model)

        self._digit_ids = [self._vocab.index(digit) for digit in "0123456789"]
        self._end_ids = [self._vocab.index(","), self._vocab.index("}")]

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
        lines: list[str] = ["Available functions:"]

        for function in self.functions:
            lines.append(self.format_function(function))
        lines.append("")
        lines.append(f"Question: {prompt}")
        lines.append('Answer: {"name": "')

        return "\n".join(lines)

    def add_text(self, input_ids: List[int], text: str) -> List[int]:
        """Return input_ids followed by the tokens of text."""
        text_ids: List[int] = self.model.encode(text)[0].tolist()
        return input_ids + text_ids

    def pick_token(
        self, ids: List[int], allowed_ids: List[int]
    ) -> int:
        """Return the allowed token the LLM prefers after ids."""
        logits = np.array(self.model.get_logits_from_input_ids(ids))

        masked = np.full_like(logits, -np.inf)
        masked[allowed_ids] = logits[allowed_ids]

        return int(np.argmax(masked))

    def _generate(
        self,
        input_ids: List[int],
        rule: Callable[[str], List[int]],
        stop_ids: List[int],
    ) -> Tuple[str, List[int]]:
        """Let the LLM write tokens allowed by rule until it stops.

        Returns the written text and input_ids with its tokens appended.
        """
        result = ""

        for _ in range(MAX_TOKENS):
            allowed_ids = rule(result)
            if not allowed_ids:
                break

            chosen = self.pick_token(input_ids, allowed_ids)
            if chosen in stop_ids:
                break

            input_ids = input_ids + [chosen]
            result = result + self._vocab[chosen]

        return result, input_ids

    def _word_rule(self, result: str, words: List[str]) -> List[int]:
        """Return the tokens that keep result on the way to a word."""
        if result in words:
            return []

        allowed_ids: List[int] = []
        for token_id, token in enumerate(self._vocab):
            if token != "" and is_start(result + token, words):
                allowed_ids.append(token_id)
        return allowed_ids

    def _number_rule(self, result: str, integer: bool) -> List[int]:
        """Return the tokens that keep result a valid number."""
        has_digit = any(char.isdigit() for char in result)
        allowed_ids = list(self._digit_ids)

        if result == "":
            allowed_ids.append(self._vocab.index("-"))
        if not integer and has_digit and "." not in result:
            allowed_ids.append(self._vocab.index("."))
        if has_digit and not result.endswith("."):
            allowed_ids += self._end_ids
        return allowed_ids

    def pick_word(
        self, input_ids: List[int], words: List[str]
    ) -> Tuple[str, List[int]]:
        """Return the LLM's pick among words and the new input_ids."""
        return self._generate(
            input_ids, lambda result: self._word_rule(result, words), []
        )

    def pick_number(
        self, input_ids: List[int], integer: bool
    ) -> Tuple[str, List[int]]:
        """Return the number the LLM writes and the new input_ids."""
        return self._generate(
            input_ids,
            lambda result: self._number_rule(result, integer),
            self._end_ids,
        )

    def pick_string(self, input_ids: List[int]) -> Tuple[str, List[int]]:
        """Return the string the LLM writes and the new input_ids."""
        quote_id = self._vocab.index('"')
        return self._generate(
            input_ids, lambda result: self._text_ids + [quote_id], [quote_id]
        )

    def _pick_parameters(
        self, input_ids: List[int], function: FunctionDefinition
    ) -> Dict[str, Union[float, int, str, bool]]:
        """Return the value the LLM writes for every parameter."""
        params: Dict[str, Union[float, int, str, bool]] = {}
        input_ids = self.add_text(input_ids, '", "parameters": {')

        for key, param in function.parameters.items():
            if params:
                input_ids = self.add_text(input_ids, ", ")
            input_ids = self.add_text(input_ids, f'"{key}": ')

            match param.type:
                case "number":
                    text, input_ids = self.pick_number(input_ids, False)
                    params[key] = float(text)
                case "integer":
                    text, input_ids = self.pick_number(input_ids, True)
                    params[key] = int(text)
                case "string":
                    input_ids = self.add_text(input_ids, '"')
                    text, input_ids = self.pick_string(input_ids)
                    input_ids = self.add_text(input_ids, '"')
                    params[key] = text
                case "boolean":
                    words = ["true", "false"]
                    text, input_ids = self.pick_word(input_ids, words)
                    params[key] = text == "true"

        return params

    def decode(self, prompt_input: PromptInput) -> FunctionCall:
        """Return the function call the LLM chooses for prompt_input."""
        text = self.build_prompt(prompt_input.prompt)
        input_ids: List[int] = self.model.encode(text)[0].tolist()

        names = [function.name for function in self.functions]
        name, input_ids = self.pick_word(input_ids, names)
        function = self.functions[names.index(name)]

        return FunctionCall(
            prompt=prompt_input.prompt,
            name=name,
            parameters=self._pick_parameters(input_ids, function),
        )
