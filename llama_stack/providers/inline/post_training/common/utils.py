
import torch
import pynvml
from typing import List

from pydantic import BaseModel

class GPUInformation(BaseModel):
    utilization: int
    power_draw: float
    total_memory: int
    used_memory: int

class SystemInformation:
    def __init__(
        self
    ) -> None:
        """Detect available GPU architecture and get memory usage."""
        gpus = None
        if torch.cuda.is_available():
           gpus = self.get_cuda_usage()
        else:
            return ["No supported GPU detected."]
        self.gpu_information = gpus


    def get_cuda_usage(self) -> tuple[list, int, int, int, int]:
        """Get GPU memory usage for NVIDIA GPUs using NVML."""
        try:
            from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, nvmlDeviceGetPowerUsage, nvmlShutdown

            nvmlInit()
            num_gpus = torch.cuda.device_count()
            gpus = []
            for i in range(num_gpus):
                handle = nvmlDeviceGetHandleByIndex(i)
                info = nvmlDeviceGetMemoryInfo(handle)
                gpu_util = nvmlDeviceGetUtilizationRates(handle).gpu  # GPU utilization in %
                power_draw = nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
                gpus.append(GPUInformation(utilization=gpu_util, power_draw=power_draw, total_memory=info.total, used_memory=info.used))
            return gpus
        except Exception as e:
            return [f"NVIDIA GPU detection failed: {str(e)}"]
        finally:
            nvmlShutdown()