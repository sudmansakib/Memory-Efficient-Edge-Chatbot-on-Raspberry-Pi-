import torch

class QuantizedKVCache:
    """
    Stores past_key_values as int8 with per-tensor scale; dequantizes on reuse.
    Only works with HF GPT2-like models that expose `past_key_values`.
    """
    def __init__(self):
        self.pkv_q = None
        self.scales = None

    @staticmethod
    def _quantize(t):
        s = t.abs().max() / 127.0 + 1e-12
        q = (t / s).round().clamp(-128, 127).to(torch.int8)
        return q, s

    @staticmethod
    def _dequantize(q, s):
        return q.float() * s

    def forward_with_cache(self, model, input_ids):
        # First pass (no cache yet)
        if self.pkv_q is None:
            out = model(input_ids=input_ids, use_cache=True)
            pkv = out.past_key_values
            q_list, s_list = [], []
            for k, v in pkv:
                kq, ks = self._quantize(k.detach().cpu())
                vq, vs = self._quantize(v.detach().cpu())
                q_list.append((kq, vq)); s_list.append((ks, vs))
            self.pkv_q, self.scales = q_list, s_list
            return out.logits
        # Subsequent pass: dequantize, append
        pkv = []
        for (kq, vq), (ks, vs) in zip(self.pkv_q, self.scales):
            k = self._dequantize(kq, ks).to(model.device)
            v = self._dequantize(vq, vs).to(model.device)
            pkv.append((k, v))
        out = model(input_ids=input_ids, past_key_values=tuple(pkv), use_cache=True)
        # refresh cache
        q_list, s_list = [], []
        for k, v in out.past_key_values:
            kq, ks = self._quantize(k.detach().cpu())
            vq, vs = self._quantize(v.detach().cpu())
            q_list.append((kq, vq)); s_list.append((ks, vs))
        self.pkv_q, self.scales = q_list, s_list
        return out.logits
