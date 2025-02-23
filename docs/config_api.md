# Configuration API Demo

## Requirements


My custom fork of LLS and the client both from the `config` branch:

llama-stack: https://github.com/cdoern/llama-stack/tree/config
llama-stack-client-python: https://github.com/cdoern/llama-stack-client-python/tree/config


## What did I add?

1. new API route: /v1/configurations
2. new inspect API: /v1/inspect/configurations
3. register endpoint for configurations: /v1/configurations/register
4. new resources in the llama-stack-client-python to handle Configurations
5. opened an issue to track my work upstream https://github.com/meta-llama/llama-stack/issues/993

## Why did I add this?

Currently there is no way for a client to

1. see the current provider configuration
2. edit the current provider after server stand-up

These gaps stood out to me, as noticable differences from what `ilab` and RHEL AI in general supports today. Having the ability for a single user system (and multi user!) to be able to see their configuration and responsibly manipulate it is important.

These two things will make it hard long term for tasks like SDG, Training, Evals, etc to be run in a repeatable and knowledgable manner by end users. Llama stack of today is built to be a "black box" to not expose sensitive configuration to the user.

Exposing the current provider configuration to a user will help them understand what they will be running for various providers as functionality gets more complex (SDG, Evals, Training, etc). Additionally, allowing a user to apply parts of a config on top of a running stack as opposed to taking the stack down and having the admin apply a full run config again seems like a more sustainable workflow.


## How can someone use this?

here is a modified version of the SDK example script which changes the configuration of a server:

```python
import os
import sys
import json
import yaml

def create_library_client(template="ollama"):
    from llama_stack import LlamaStackAsLibraryClient

    # I am manually passing in one of my paths here.
    client = LlamaStackAsLibraryClient("/Users/charliedoern/.llama/distributions/ollama/ollama-run.yaml")
    client.initialize()
    return client


client = (
    create_library_client()
)

prov = client.providers.list()
config = client.configurations.inspect()

print("Old Config \n", yaml.dump(config, indent=2))


# put together a new config. Can also be read from a file
# important point here is that this can be a partial provider config list.

config = { "inference": [{'provider_id': 'ollama', 'provider_type': 'remote::ollama', 'config': {'url': 'http://localhost:12345'}}]}
config = json.dumps(config)


config = client.configurations.register(config=config)


# get new configuration

config = client.configurations.inspect()
try:
    print("New Configuration \n", yaml.dump(config, indent=2))
except Exception as exc:
    print(f"could not dump yaml: {exc}")


# List available models

try:
    models = client.models.list()
except AttributeError as e:
    print(e)
    sys.exit(1)
print("--- Available models: ---")
for m in models:
    print(f"- {m.identifier}")
print()

# uses the NEW ollama URL not the old one.

response = client.inference.chat_completion(
    model_id=os.environ["INFERENCE_MODEL"],
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about coding"},
    ],
)
```


This script switches the ollama URL of an existing server to point to another ollama server. 

## Some Key Features

### UserConfig vs StackRunConfig

A key part of this API are the fields exposed in both the inspection and registration. A Configuration object contains a StackRunConfig within it. However, the data within this config is a UserConfig. For those unaware, a `StackRunConfig` generally follow the structure of the run.yaml a user specifies when running `llama stack run`.

A UserConfig is a StackRunConfig but only with specific fields displayed to the user. Since each provider has its own config class that feeds into the StackRunConfig the following can be used to label certain fields as "User Configurable":

```python
class OllamaImplConfig(BaseModel):
    url: str = Field(DEFAULT_OLLAMA_URL, json_schema_extra={"user_field": True})

    @classmethod
    def sample_run_config(cls, url: str = "${env.OLLAMA_URL:http://localhost:11434}", **kwargs) -> Dict[str, Any]:
        return {"url": url}
```

the pydantic json_schema_extra field can then be used when creating a Configuration object to create an intermediary UserConfig. The User Config will only have fields labeled as user_field meaning that if a user tries to register a configuration with non-user fields specified, they will be dropped, and an inspected configuration will only contain user fields for viewing as well. This structure results in `client.configurations.inspect()` output like:

```
New Configuration:
 providers:
  agents:
  - config: {}
    provider_id: meta-reference
    provider_type: inline::meta-reference
  datasetio: []
  eval: []
  inference:
  - config:
      url: http://localhost:12345
    provider_id: ollama
    provider_type: remote::ollama
  safety: []
  scoring:
  - config: {}
    provider_id: braintrust
    provider_type: inline::braintrust
  telemetry:
  - config: {}
    provider_id: meta-reference
    provider_type: inline::meta-reference
  tool_runtime:
  - config: {}
    provider_id: brave-search
    provider_type: remote::brave-search
  - config: {}
    provider_id: tavily-search
    provider_type: remote::tavily-search
  vector_io:
  - config: {}
    provider_id: sqlite-vec
    provider_type: inline::sqlite-vec
```

So far, only the ollama URL has been added as a valid UserConfig field, meaning its the only provider config that shows up.

## Summary

the new Configuration API and expanded Inspect API allow a user to see parts of the stack configuration specifically, permitted parts of their provider configs. Additionally, a user can apply these corresponding fields in a "register" method that can update a stack configuration in place without needing to take down the server.