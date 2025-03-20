# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Literal, Optional
from kfp import dsl

from pydantic import BaseModel


class InstructLabKubeFlowPostTrainingConfig(BaseModel):
    experiment_name: str
    skills_processed_data: dsl.Output[dsl.Dataset]
    skills_pvc_path: str = "/data/skills"
    knowledge_processed_data: dsl.Output[dsl.Dataset]
    knowledge_pvc_path: str = "/data/knowledge"
    torch_seed: Optional[int] = None
    model_path: str = "/model"
    sdg_path: str = "/data/sdg"
    skills_path: str = "/data/skills"
    knowledge_path: str = "/data/knowledge"
    gpu_identifier: str
    cpu_per_worker: str
    memory_per_worker: str
    tolerations: list
    node_selectors: dict
    pytorchjob_output_yaml: dsl.Output[dsl.Artifact]
    model_pvc_name: str
    input_pvc_name: str
    output_pvc_name: str
    name_suffix: str
    phase_num: int
    base_image: str
    nproc_per_node: int = 3
    num_warmup_steps: int = 800
    save_samples: int = 0
    seed: int = 42
    job_timeout: int = 86400
    delete_after_done: bool = False
    checkpoint_format: Optional[Literal["meta", "huggingface"]] = "meta"