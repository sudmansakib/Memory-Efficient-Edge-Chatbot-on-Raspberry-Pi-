from typing import List, Optional, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class HFBackend:
    """
    Pi-friendly HuggingFace backend with optional QuantizedKVCache support.
    CPU-only execution.
    Manual token loop exposes past_key_values for compression.
    """

    def __init__(
        self,
        model_name: str = "distilgpt2",
        max_ctx: int = 512,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.max_ctx = max_ctx

        # Always CPU on Pi; allow override for laptop.
        self.device = device or "cpu"


        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        torch.set_grad_enabled(False)

    # --- token utilities -------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def token_count(self, text: str) -> int:
        return len(self.encode(text))

    # --- sampling util ---------------------------------------------------------

    @staticmethod
    def _sample_next(logits: torch.Tensor, temperature: float, top_p: float) -> int:
        if temperature <= 0.0:
            return int(torch.argmax(logits).item())

        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)

        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)

        mask = cumsum <= top_p
        mask[..., 0] = True  # always keep top

        filtered = sorted_probs * mask
        filtered = filtered / filtered.sum()

        choice = torch.multinomial(filtered, 1).item()
        return int(sorted_idx[choice].item())

    # --- NEW: custom generation with quantized KV-cache ------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_p: float = 0.95,
        kv_cache: Optional[Any] = None,   # <-- REQUIRED PARAM
    ) -> str:

        # Tokenize prompt
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_ctx
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        # 1) Initial forward pass
        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]
            past_kv = outputs.past_key_values

            if hasattr(kv_cache, "store"):
                kv_cache.store(past_kv)


        generated_ids = []

        # 2) Token-by-token loop
        for _ in range(max_new_tokens):

            # Retrieve quantized KV if used
            if hasattr(kv_cache, "get"):
                past_kv = kv_cache.get()

                if past_kv is None:
                    break

            # next token
            next_id = self._sample_next(
                logits[0], temperature=temperature, top_p=top_p
            )
            generated_ids.append(next_id)

            # feed next token
            next_input = torch.tensor([[next_id]], device=self.device)

            with torch.inference_mode():
                outputs = self.model(
                    input_ids=next_input,
                    past_key_values=past_kv,
                    use_cache=True,
                )
                logits = outputs.logits[:, -1, :]
                past_kv = outputs.past_key_values

                if hasattr(kv_cache, "store"):
                    kv_cache.store(past_kv)


        return self.decode(generated_ids)
