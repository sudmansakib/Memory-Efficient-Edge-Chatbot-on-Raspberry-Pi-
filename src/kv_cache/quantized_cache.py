import torch
from typing import Any, List, Tuple, Optional

class QuantizedKVCache:
    """
    INT8 quantized KV-cache.
    Stores quantized past_key_values and dequantizes before reuse.
    """

    def __init__(self):
        self.store_q: Optional[
            List[List[Tuple[torch.Tensor, float]]]
        ] = None  # layers -> [(q_tensor, scale), ...]

    @staticmethod
    def quantize_tensor(t: torch.Tensor) -> Tuple[torch.Tensor, float]:
        max_val = t.abs().max().item()
        if max_val == 0:
            return torch.zeros_like(t, dtype=torch.int8), 1.0
        scale = max_val / 127.0
        q = (t / scale).round().clamp(-128, 127).to(torch.int8)
        return q, scale

    @staticmethod
    def dequantize_tensor(q: torch.Tensor, scale: float) -> torch.Tensor:
        return q.float() * scale

    def store(self, past_key_values: Any) -> None:
        layers_q = []
        for layer in past_key_values:
            k, v = layer
            qk, sk = self.quantize_tensor(k.cpu())
            qv, sv = self.quantize_tensor(v.cpu())
            layers_q.append([(qk, sk), (qv, sv)])
        self.store_q = layers_q

    def get(self) -> Optional[Any]:
        if self.store_q is None:
            return None

        restored = []
        for layer in self.store_q:
            (qk, sk), (qv, sv) = layer
            k = self.dequantize_tensor(qk, sk)
            v = self.dequantize_tensor(qv, sv)
            restored.append((k, v))
        return tuple(restored)
