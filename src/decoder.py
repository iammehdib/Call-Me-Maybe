import json
from typing import ClassVar, Dict, List

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
