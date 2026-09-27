from typing import Dict, Literal, Union
from pydantic import BaseModel

ParamTypeName = Literal["number", "integer", "string", "boolean"]

class ParamType(BaseModel):
    """Type declaration of a parameter or return value."""

    type: ParamTypeName

class FunctionDefinition(BaseModel):
    """A callable function the LLM can choose from."""

    name: str
    description: str
    parameters: Dict[str, ParamType]
    returns: ParamType

class PromptInput(BaseModel):
    """A natural language request to translate."""

    prompt: str

class FunctionCall(BaseModel):
    """The function call produced for one prompt."""

    prompt: str
    name: str
    parameters: Dict[str, Union[float, int, str, bool]]