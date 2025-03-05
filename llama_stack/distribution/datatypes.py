# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from llama_stack.distribution.utils.dynamic import instantiate_class_type

from llama_stack.apis.benchmarks import Benchmark, BenchmarkInput
from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Dataset, DatasetInput
from llama_stack.apis.eval import Eval
from llama_stack.apis.inference import Inference
from llama_stack.apis.models import Model, ModelInput
from llama_stack.apis.safety import Safety
from llama_stack.apis.scoring import Scoring
from llama_stack.apis.scoring_functions import ScoringFn, ScoringFnInput
from llama_stack.apis.shields import Shield, ShieldInput
from llama_stack.apis.tools import Tool, ToolGroup, ToolGroupInput, ToolRuntime
from llama_stack.apis.vector_dbs import VectorDB, VectorDBInput
from llama_stack.apis.vector_io import VectorIO
from llama_stack.providers.datatypes import Api, ProviderSpec
from llama_stack.providers.utils.kvstore.config import KVStoreConfig

LLAMA_STACK_BUILD_CONFIG_VERSION = "2"
LLAMA_STACK_RUN_CONFIG_VERSION = "2"





# Example: /inference, /safety
class AutoRoutedProviderSpec(ProviderSpec):
    provider_type: str = "router"
    config_class: str = ""

    container_image: Optional[str] = None
    routing_table_api: Api
    module: str
    provider_data_validator: Optional[str] = Field(
        default=None,
    )

    @property
    def pip_packages(self) -> List[str]:
        raise AssertionError("Should not be called on AutoRoutedProviderSpec")


# Example: /models, /shields
class RoutingTableProviderSpec(ProviderSpec):
    provider_type: str = "routing_table"
    config_class: str = ""
    container_image: Optional[str] = None

    router_api: Api
    module: str
    pip_packages: List[str] = Field(default_factory=list)


class DistributionSpec(BaseModel):
    description: Optional[str] = Field(
        default="",
        description="Description of the distribution",
    )
    container_image: Optional[str] = None
    providers: Dict[str, Union[str, List[str]]] = Field(
        default_factory=dict,
        description="""
Provider Types for each of the APIs provided by this distribution. If you
select multiple providers, you should provide an appropriate 'routing_map'
in the runtime configuration to help route to the correct provider.""",
    )


class Provider(BaseModel):
    provider_id: str
    provider_type: str
    config: Dict[str, Any]

class ServerConfig(BaseModel):
    port: int = Field(
        default=8321,
        description="Port to listen on",
        ge=1024,
        le=65535,
    )
    tls_certfile: Optional[str] = Field(
        default=None,
        description="Path to TLS certificate file for HTTPS",
    )
    tls_keyfile: Optional[str] = Field(
        default=None,
        description="Path to TLS key file for HTTPS",
    )


class UserConfig(BaseModel):
    providers: Dict[str, List[Provider]] = Field(
        description="""
One or more providers to use for each API. The same provider_type (e.g., meta-reference)
can be instantiated multiple times (with different configs) if necessary.
""",
    )
    @classmethod
    def from_stack_run(cls, registry: Dict[Any, Dict[str, Any]], stack_run: "StackRunConfig") -> "UserConfig":
        """
        This is almost a method to go backwards and get a user config from an existing run config
        """
        user_config : Dict[str, List[Provider]] = {}
        for type, providers in stack_run.providers.items():
            api = Api(type)
            user_config[type] = []
            for provider in providers:
                provider_config = {}
                provider_spec = registry[api][provider.provider_type]
                config_type = instantiate_class_type(provider_spec.config_class)
                try:
                    if provider.config:
                        existing = config_type(**provider.config)
                        for field_name, field in existing.model_fields.items():
                                if field.json_schema_extra:
                                    provider_config[field_name] = field.default
                        user_config[type].append(Provider(provider_id=provider.provider_id, provider_type=provider.provider_type, config=provider_config))
                except Exception as exc:
                    print(f"Could not instantiate UserConfig due to improper provider config {exc}")           
        return cls(providers=user_config)
    
    @classmethod
    def from_providers(cls, registry: Dict[Any, Dict[str, Any]], providers: Dict[str, List[Provider]]):
        """
        This is a method to go forward, validate that a dictionary of providers is _only_ a user config
        """
        user_config : Dict[str, List[Provider]] = {}
        for type, provider_list in providers.items():
            api = Api(type)
            user_config[type] = []
            provider_config = {}
            for prov in provider_list:
                prov = Provider(**prov)
                provider_spec = registry[api][prov.provider_type]
                config_type = instantiate_class_type(provider_spec.config_class)
                try:
                    if prov.config is not None:
                        existing = config_type(**prov.config)
                        for field_name, field in existing.model_fields.items():
                                if field.json_schema_extra:
                                    provider_config[field_name] =  getattr(existing, field_name)
                                else:
                                    print("given configuration is not user configurable.")
                        user_config[type].append(Provider(provider_id=prov.provider_id, provider_type=prov.provider_type, config=provider_config))
                except Exception as exc:
                    print(f"Could not instantiate UserConfig due to improper provider config {exc}")           
        return cls(providers=user_config)

class StackRunConfig(BaseModel):
    version: str = LLAMA_STACK_RUN_CONFIG_VERSION

    image_name: str = Field(
        ...,
        description="""
Reference to the distribution this package refers to. For unregistered (adhoc) packages,
this could be just a hash
""",
    )
    container_image: Optional[str] = Field(
        default=None,
        description="Reference to the container image if this package refers to a container",
    )
    apis: List[str] = Field(
        default_factory=list,
        description="""
The list of APIs to serve. If not specified, all APIs specified in the provider_map will be served""",
    )

    providers: Dict[str, List[Provider]] = Field(
        description="""
One or more providers to use for each API. The same provider_type (e.g., meta-reference)
can be instantiated multiple times (with different configs) if necessary.
""",
    )
    metadata_store: Optional[KVStoreConfig] = Field(
        default=None,
        description="""
Configuration for the persistence store used by the distribution registry. If not specified,
a default SQLite store will be used.""",
    )

    # registry of "resources" in the distribution
    models: List[ModelInput] = Field(default_factory=list)
    shields: List[ShieldInput] = Field(default_factory=list)
    vector_dbs: List[VectorDBInput] = Field(default_factory=list)
    datasets: List[DatasetInput] = Field(default_factory=list)
    scoring_fns: List[ScoringFnInput] = Field(default_factory=list)
    benchmarks: List[BenchmarkInput] = Field(default_factory=list)
    tool_groups: List[ToolGroupInput] = Field(default_factory=list)

    server: ServerConfig = Field(
        default_factory=ServerConfig,
        description="Configuration for the HTTP(S) server",
    )


class BuildConfig(BaseModel):
    version: str = LLAMA_STACK_BUILD_CONFIG_VERSION

    distribution_spec: DistributionSpec = Field(description="The distribution spec to build including API providers. ")
    image_type: str = Field(
        default="conda",
        description="Type of package to build (conda | container | venv)",
    )
