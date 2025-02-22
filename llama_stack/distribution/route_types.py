from pydantic import BaseModel, Field

from typing import Annotated, Any, Dict, List, Optional, Union

from llama_stack.apis.config import Configuration
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


RoutingKey = Union[str, List[str]]


RoutableObject = Union[
    Configuration,
    Model,
    Shield,
    VectorDB,
    Dataset,
    ScoringFn,
    Benchmark,
    Tool,
    ToolGroup,
]


RoutableObjectWithProvider = Annotated[
    Union[
        Configuration,
        Model,
        Shield,
        VectorDB,
        Dataset,
        ScoringFn,
        Benchmark,
        Tool,
        ToolGroup,
    ],
    Field(discriminator="type"),
]

RoutedProtocol = Union[
    Inference,
    Safety,
    VectorIO,
    DatasetIO,
    Scoring,
    Eval,
    ToolRuntime,
]