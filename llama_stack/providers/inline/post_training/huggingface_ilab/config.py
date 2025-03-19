# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Literal, Optional

from pydantic import BaseModel

MINIMUM_VRAM_REQUIREMENT = 9.6e10 # 96 GB?
GPU_MAXIMUM_POWER_DRAW = 72 # 72w maximum power draw?
GPU_MAXIMUM_UTILIZATION = 20 # if 20% of GPU is already bring used percentage wise, error.

class HFilabPostTrainingConfig(BaseModel):
    torch_seed: Optional[int] = None
    checkpoint_format: Optional[Literal["meta", "huggingface"]] = "meta"
