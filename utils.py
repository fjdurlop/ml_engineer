import argparse
import math
import time
import statistics as stats
from dataclasses import dataclass
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Quantization / precision helpers
# ----------------------------
def apply_int8_dynamic_cpu(model: nn.Module) -> nn.Module:
    """
    Dynamic quantization for CPU on supported modules (Linear, LSTM/GRU).
    """
    quantized = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM, nn.GRU},
        dtype=torch.qint8,
    )
    return quantized

@dataclass
class DevicePolicy:
    device_str: str        # "cpu" or "cuda:0"
    precision: str         # "fp32" | "fp16" | "bf16" | "int8-cpu"
    autocast_dtype: Optional[torch.dtype]  # torch.float16 or torch.bfloat16 or None
    

def detect_auto_policy(device_arg: str) -> DevicePolicy:
    """
    Heterogeneous policy:
      - If CUDA:
          * Ampere+ (sm_80/90): prefer BF16 (stable numerics)
          * Volta/Turing (sm_70/75): FP16
      - Else CPU: INT8 dynamic
    """
    if device_arg == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda:0")
            major, minor = torch.cuda.get_device_capability(dev)
            print(f"major: {major} minor: {minor}")
            if major >= 8:     # Ampere or newer  bfloat16 is great
                return DevicePolicy("cuda:0", "bf16", torch.bfloat16)
            else:              # Volta (7.0) / Turing (7.5)  fp16 sweet spot
                return DevicePolicy("cuda:0", "fp16", torch.float16)
        else:
            return DevicePolicy("cpu", "int8-cpu", None)
    else:
        # Non-auto device specified; default to fp32 unless invalid for CUDA
        if device_arg.startswith("cuda"):
            return DevicePolicy(device_arg, "fp16", torch.float16)
        return DevicePolicy(device_arg, "int8-cpu", None)

def prepare_model_for_precision(model: nn.Module, policy: DevicePolicy) -> nn.Module:
    """
    Moves and converts model to the requested precision.
    """
    print(f"prepare_model_for_precision: policy: {policy}")
    print(f"prepare_model_for_precision: policy.precision: {policy.precision}")
    
    
    if policy.device_str.startswith("cuda"):
        device = torch.device(policy.device_str)
        model = model.to(device)
        if policy.precision == "fp16":
            print(f"prepare_model_for_precision: model.half()")
            model = model.half()
            # ensure parameters in fp16, check forward torch ops are fp16 compatible
        elif policy.precision == "bf16":
            # Keep module weights as bf16 where supported; fall back to autocast otherwise
            # Many RNNs are ok under autocast; we still place to CUDA device.
            for p in model.parameters():
                p.data = p.data.to(torch.bfloat16)
        # fp32 default already handled
    else:
        # CPU path
        if policy.precision == "int8-cpu":
            model = apply_int8_dynamic_cpu(model)
        # else fp32 CPU
    model.eval()
    return model, policy