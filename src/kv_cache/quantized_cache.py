# src/kv_cache/quantized_cache.py

import torch
from typing import List, Tuple


class QuantizedKVCache:
    """
    Skeleton for a Quantized KV-cache.
    - Quantizes tensors to int8 with a single scale per tensor.
    - Stores (q_tensor, scale) pairs.
    You can integrate this with HF past_key_values when you’re ready.
    """

    def __init__(self) -> None:
        self.kv_q: List[Tuple[torch.Tensor, float]] | None = None

    @staticmethod
    def quantize(t: torch.Tensor) -> Tuple[torch.Tensor, float]:
        max_val = t.abs().max().item()
        if max_val == 0:
            scale = 1.0
        else:
            scale = max_val / 127.0
        q = (t / scale).round().clamp(-128, 127).to(torch.int8)
        return q, scale

    @staticmethod
    def dequantize(q: torch.Tensor, scale: float) -> torch.Tensor:
        return q.to(torch.float32) * scale

    def clear(self) -> None:
        self.kv_q = None
