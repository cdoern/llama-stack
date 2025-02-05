from pydantic import BaseModel, Field
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    runtime_checkable,
    Union,
)
from llama_stack.apis.resource import Resource, ResourceType

from llama_models.schema_utils import json_schema_type, register_schema, webmethod
from llama_stack.providers.utils.telemetry.trace_protocol import trace_protocol


@json_schema_type
class Configurartion(BaseModel):
    type: Literal[ResourceType.configuration.value] = ResourceType.configuration.value
    data: dict[str, Any]

class ConfigListResponse(BaseModel):
    data: List[dict[str, Any]]


@runtime_checkable
@trace_protocol
class Configurations(Protocol):
    """Llama Stack Configuration API for storing and applying hyperparameters for given tasks.
    
    """
    @webmethod(route="/configurations", method="GET")
    async def list_configs(
        self,
    ) -> Optional[ConfigListResponse]: ...

    @webmethod(route="/configurations/{config_id}", method="GET")
    async def get_config(
        self,
        config_id,
    ) -> Optional[Configurartion]: ...

    @webmethod(route="/configurations", method="POST")
    async def register_config(
        self,
        config_id,
        provider_config_id,
        provider_id,
    ) -> dict[str, Any]: ...