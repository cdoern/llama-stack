# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import logging
import textwrap

from pydantic import BaseModel
from typing import Any, Dict

from llama_stack.apis.config import (
    Configurations,
    ConfigListResponse,
    Configuration,
)

from llama_stack.apis.inspect import (
    InspectConfigResponse
)

import copy

from ruamel.yaml import YAML

from llama_stack.distribution.routers.routing_tables import register_object_with_provider, RoutableObjectWithProvider

from llama_stack.distribution.datatypes import (
    LLAMA_STACK_RUN_CONFIG_VERSION,
    DistributionSpec,
    Provider,
    StackRunConfig,
    UserConfig
)
from llama_stack.distribution.distribution import (
    builtin_automatically_routed_apis,
    get_provider_registry,
)
from llama_stack.distribution.utils.dynamic import instantiate_class_type
from llama_stack.distribution.utils.prompt_for_config import prompt_for_config
from llama_stack.providers.datatypes import Api, ProviderSpec

logger = logging.getLogger(__name__)


from typing import Annotated, Any, Dict, List, Optional, Union


class Config(BaseModel):
    run_config: StackRunConfig


async def get_provider_impl(config, deps):
    impl = ConfigImpl(config, deps)
    await impl.initialize()
    return impl

# is this technically "server side"
# but if the config files are going to live here, that means they are "server side"
# distributions are server side so shouldn't config for distributions be server side?
class ConfigImpl(Configurations):
    def __init__(self, config, deps):
        self.config = config
        self.deps = deps

    def dictionary_to_provider_config(config_path: str) -> UserConfig:
        # read file
        # render a userconfig from file. 
        # need to be able to specify a partial config of say 1 full provider minumum. So what we should
        # be expecting is a json file full of provider configs. So map into a list of providers and then add a new
        # user config func to go from list of providers to user cfg
        yaml = YAML()
        yaml.indent(mapping=2, sequence=4, offset=2)
        provider_registry = get_provider_registry()
        with open(config_path, "r", encoding="utf-8") as yamlfile:
                    content = yaml.load(yamlfile)
                    if isinstance(content, dict):
                        return UserConfig.from_providers(registry=provider_registry, providers=Dict[str, List[Provider]](**content))
                        

    def merge_dicts(self, base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merges `overrides` into `base`, replacing only specified keys."""
        merged = copy.deepcopy(base)  # Preserve original dict
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                # Recursively merge if both are dictionaries
                merged[key] = self.merge_dicts(merged[key], value)
            else:
                # Otherwise, directly override
                merged[key] = value
        
        return merged

    def merge_configs(self, global_config: Dict[str, List[Provider]], new_config: Dict[str, List[Provider]]) -> Dict[str, List[Provider]]:
        merged_config = copy.deepcopy(global_config)  # Preserve original structure

        for key, new_providers in new_config.items():
            if key in merged_config:
                existing_providers = {p.provider_id: p for p in merged_config[key]}

                for new_provider in new_providers:
                    if new_provider.provider_id in existing_providers:
                        # Override settings of existing provider
                        existing = existing_providers[new_provider.provider_id]
                        existing.config = self.merge_dicts(existing.config, new_provider.config)
                    else:
                        # Append new provider
                        merged_config[key].append(new_provider)
            else:
                # Add new category entirely
                merged_config[key] = new_providers

        return merged_config

    async def register_config(
         self,
         config,
      #   config: str,
     ) -> Configuration:
        # get current user config from stack config
        # see if there are any registered configs
        # apply those on top
        # see if this one collides, if so err, if not apply
        provider_registry = get_provider_registry()

        # do this just to validate that whas is being given is a user config
        import ast
        config = ast.literal_eval(config)
        prov = dict[str, List[Provider]](**config)
        user_config = UserConfig.from_providers(registry=provider_registry, providers=prov)

        # map user config ON TOP OF run config
        # register config with provider
        new_config = self.merge_configs(self.config.run_config.providers, user_config.providers)
        self.config.run_config.providers = new_config

        # hmmmm we might need more than `data`. We havent actually used the `Configuration` obj yet, might need some reshaping
        config = Configuration(
            config=self.config.run_config
        )

        return config

        #user_config = self.dictionary_to_provider_config(config_path)
       # user_config = UserConfig.from_stack_run(registry=provider_registry, stack_run=run_config)
       # config = Configuration(
    #
    #    )
        # we now have a read config into a user config, need to add
        # provider_id(s) to user_config probably such that it is clear which providers it is meant to apply to
        # we could pull this out of the config, but users might want to just apply the parts of it to specific providers
       # registered_config = await self.register_object()
        # one config per provider per client. You don't need to check if the configs overlap, just if the providers you are trying
        # to apply them to overlap

    
    async def register_object(self, obj: RoutableObjectWithProvider) -> Configuration:
        # the actual impl of this registers a cfg with the providers

        # each provider needs a register_config func



        # if provider_id is not specified, pick an arbitrary one from existing entries
        if not obj.provider_id and len(self.impls_by_provider_id) > 0:
            obj.provider_id = list(self.impls_by_provider_id.keys())[0]

        if obj.provider_id not in self.impls_by_provider_id:
            raise ValueError(f"Provider `{obj.provider_id}` not found")

        p = self.impls_by_provider_id[obj.provider_id]

        registered_obj = await register_object_with_provider(obj, p)
        await self.dist_registry.register(registered_obj)
        return registered_obj



    async def initialize(self) -> None:
        pass

    async def list_configs(self) -> ConfigListResponse:
        pass

    async def get_config(self, config_id) -> Configuration:
        pass


def configure_single_provider(registry: Dict[str, ProviderSpec], provider: Provider) -> Provider:
    provider_spec = registry[provider.provider_type]
    config_type = instantiate_class_type(provider_spec.config_class)
    try:
        if provider.config:
            existing = config_type(**provider.config)
        else:
            existing = None
    except Exception:
        existing = None

    cfg = prompt_for_config(config_type, existing)
    return Provider(
        provider_id=provider.provider_id,
        provider_type=provider.provider_type,
        config=cfg.dict(),
    )


def configure_api_providers(config: StackRunConfig, build_spec: DistributionSpec) -> StackRunConfig:
    is_nux = len(config.providers) == 0

    if is_nux:
        logger.info(
            textwrap.dedent(
                """
        Llama Stack is composed of several APIs working together. For each API served by the Stack,
        we need to configure the providers (implementations) you want to use for these APIs.
"""
            )
        )

    provider_registry = get_provider_registry()
    builtin_apis = [a.routing_table_api for a in builtin_automatically_routed_apis()]

    if config.apis:
        apis_to_serve = config.apis
    else:
        apis_to_serve = [a.value for a in Api if a not in (Api.telemetry, Api.inspect)]

    for api_str in apis_to_serve:
        api = Api(api_str)
        if api in builtin_apis:
            continue
        if api not in provider_registry:
            raise ValueError(f"Unknown API `{api_str}`")

        existing_providers = config.providers.get(api_str, [])
        if existing_providers:
            logger.info(
                f"Re-configuring existing providers for API `{api_str}`...",
                "green",
                attrs=["bold"],
            )
            updated_providers = []
            for p in existing_providers:
                logger.info(f"> Configuring provider `({p.provider_type})`")
                updated_providers.append(configure_single_provider(provider_registry[api], p))
                logger.info("")
        else:
            # we are newly configuring this API
            plist = build_spec.providers.get(api_str, [])
            plist = plist if isinstance(plist, list) else [plist]

            if not plist:
                raise ValueError(f"No provider configured for API {api_str}?")

            logger.info(f"Configuring API `{api_str}`...", "green", attrs=["bold"])
            updated_providers = []
            for i, provider_type in enumerate(plist):
                if i >= 1:
                    others = ", ".join(plist[i:])
                    logger.info(
                        f"Not configuring other providers ({others}) interactively. Please edit the resulting YAML directly.\n"
                    )
                    break

                logger.info(f"> Configuring provider `({provider_type})`")
                updated_providers.append(
                    configure_single_provider(
                        provider_registry[api],
                        Provider(
                            provider_id=(f"{provider_type}-{i:02d}" if len(plist) > 1 else provider_type),
                            provider_type=provider_type,
                            config={},
                        ),
                    )
                )
                logger.info("")

        config.providers[api_str] = updated_providers

    return config


def upgrade_from_routing_table(
    config_dict: Dict[str, Any],
) -> Dict[str, Any]:
    def get_providers(entries):
        return [
            Provider(
                provider_id=(f"{entry['provider_type']}-{i:02d}" if len(entries) > 1 else entry["provider_type"]),
                provider_type=entry["provider_type"],
                config=entry["config"],
            )
            for i, entry in enumerate(entries)
        ]

    providers_by_api = {}

    routing_table = config_dict.get("routing_table", {})
    for api_str, entries in routing_table.items():
        providers = get_providers(entries)
        providers_by_api[api_str] = providers

    provider_map = config_dict.get("api_providers", config_dict.get("provider_map", {}))
    if provider_map:
        for api_str, provider in provider_map.items():
            if isinstance(provider, dict) and "provider_type" in provider:
                providers_by_api[api_str] = [
                    Provider(
                        provider_id=f"{provider['provider_type']}",
                        provider_type=provider["provider_type"],
                        config=provider["config"],
                    )
                ]

    config_dict["providers"] = providers_by_api

    config_dict.pop("routing_table", None)
    config_dict.pop("api_providers", None)
    config_dict.pop("provider_map", None)

    config_dict["apis"] = config_dict["apis_to_serve"]
    config_dict.pop("apis_to_serve", None)

    return config_dict


def parse_and_maybe_upgrade_config(config_dict: Dict[str, Any]) -> StackRunConfig:
    version = config_dict.get("version", None)
    if version == LLAMA_STACK_RUN_CONFIG_VERSION:
        return StackRunConfig(**config_dict)

    if "routing_table" in config_dict:
        logger.info("Upgrading config...")
        config_dict = upgrade_from_routing_table(config_dict)

    config_dict["version"] = LLAMA_STACK_RUN_CONFIG_VERSION

    return StackRunConfig(**config_dict)
