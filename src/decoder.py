import json
from typing import ClassVar, Dict, List

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr

from llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition


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

    def pick_token(self, ids: List[int], allowed: List[int]) -> int:
        """Return the allowed token the LLM prefers after ids."""
        logits = np.array(self.model.get_logits_from_input_ids(ids))

        masked = np.full_like(logits, -np.inf)
        masked[allowed] = logits[allowed]

        return int(np.argmax(masked))
