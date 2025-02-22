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
from llama_stack.distribution.datatypes import StackRunConfig
from llama_stack.apis.resource import Resource, ResourceType

from llama_stack.schema_utils import json_schema_type, register_schema, webmethod
from llama_stack.providers.utils.telemetry.trace_protocol import trace_protocol


@json_schema_type
class Configuration(BaseModel):
    type: Literal[ResourceType.configuration.value] = ResourceType.configuration.value
    config: StackRunConfig

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
    ) -> Optional[Configuration]: ...

    @webmethod(route="/configurations/register", method="POST")
    async def register_config(
        self,
        config,
    ) -> dict[str, Any]: ...