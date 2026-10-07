from typing import Annotated, Dict, Literal, Union
from pydantic import BaseModel, StringConstraints

# A valid identifier: letters, digits and "_", not empty, no leading digit.
Name = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]


class ParamType(BaseModel):
    """Type declaration of a parameter or return value."""

    type: Literal["number", "integer", "string", "boolean"]


class FunctionDefinition(BaseModel):
    """A callable function the LLM can choose from."""

    name: Name
    description: str
    parameters: Dict[Name, ParamType]
    returns: ParamType


class PromptInput(BaseModel):
    """A natural language request to translate."""

    prompt: str


class FunctionCall(BaseModel):
    """The function call produced for one prompt."""

    prompt: str
    name: str
    parameters: Dict[str, Union[float, int, str, bool]]
