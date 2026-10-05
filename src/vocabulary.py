import json
from typing import Dict, List

from pydantic import BaseModel

from llm_sdk import Small_LLM_Model

texts_cache: Dict[str, List[str]] = {}


def read_texts(model: Small_LLM_Model) -> List[str]:
    """Return the text of every token of model, read once per vocab file."""
    path = model.get_path_to_vocab_file()

    if path not in texts_cache:
        with open(path, "r", encoding="utf-8") as f:
            token_to_id: Dict[str, int] = json.load(f)

        texts: List[str] = []
        for token_id in range(len(token_to_id)):
            texts.append(model.decode([token_id]))
        texts_cache[path] = texts

    return texts_cache[path]


class Vocabulary(BaseModel):
    """Text of every token and the token ids the decoder uses."""

    texts: List[str]
    digit_ids: List[int]
    end_ids: List[int]
    space_id: int
    space_minus_id: int
    dot_id: int
    text_ids: List[int]
    quote_ids: List[int]

    def __init__(self, model: Small_LLM_Model) -> None:
        """Find the token ids the decoder needs in the vocab of model."""
        texts = read_texts(model)
        text_ids: List[int] = []
        quote_ids: List[int] = []

        for token_id, token in enumerate(texts):
            if '"' in token:
                quote_ids.append(token_id)
            elif token != "" and "\n" not in token:
                text_ids.append(token_id)

        super().__init__(
            texts=texts,
            digit_ids=[texts.index(digit) for digit in "0123456789"],
            end_ids=[texts.index(","), texts.index("}")],
            space_id=texts.index(" "),
            space_minus_id=texts.index(" -"),
            dot_id=texts.index("."),
            text_ids=text_ids,
            quote_ids=quote_ids,
        )
