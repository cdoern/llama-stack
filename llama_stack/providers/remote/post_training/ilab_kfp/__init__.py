# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import InstructLabImplConfig


async def get_adapter_impl(config: InstructLabImplConfig, _deps):
    from .instructlab_kfp import InstructLabPostTrainingImpl

    impl = InstructLabPostTrainingImpl(config.url)
    await impl.initialize()
    return impl
