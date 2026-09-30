import json
from typing import ClassVar, Dict, List, Tuple, Union

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

    def __init__(
        self, model: Small_LLM_Model, functions: List[FunctionDefinition]
    ) -> None:
        """Validate the fields, then get the vocabulary of model."""
        super().__init__(model=model, functions=functions)
        self._vocab = self.get_vocab(model)

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

    def pick_token(
        self, ids: List[int], allowed_ids: List[int]
    ) -> int:
        """Return the allowed token the LLM prefers after ids."""
        logits = np.array(self.model.get_logits_from_input_ids(ids))

        masked = np.full_like(logits, -np.inf)
        masked[allowed_ids] = logits[allowed_ids]

        return int(np.argmax(masked))

    def pick_word(
        self, input_ids: List[int], words: List[str]
    ) -> Tuple[str, List[int]]:
        """Return the LLM's pick among words and the new input_ids."""
        result = ""

        for _ in range(MAX_TOKENS):
            allowed_ids: List[int] = []
            for token_id, token in enumerate(self._vocab):
                candidate_word = result + token
                if token != "" and is_start(candidate_word, words):
                    allowed_ids.append(token_id)

            chosen = self.pick_token(input_ids, allowed_ids)
            input_ids = input_ids + [chosen]
            result = result + self._vocab[chosen]

            if result in words:
                break

        return result, input_ids

    def _pick_parameters(
        self, function: FunctionDefinition
    ) -> Dict[str, Union[float, int, str, bool]]:
        """Return a value for every parameter of function."""
        params: Dict[str, Union[float, int, str, bool]] = {}

        for key, param in function.parameters.items():
            match param.type:
                case "number":
                    params[key] = 0.0
                case "integer":
                    params[key] = 0
                case "string":
                    params[key] = ""
                case "boolean":
                    params[key] = False

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
            parameters=self._pick_parameters(function)
        )
