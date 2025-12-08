# src/inference/hf_backend_pi.py
#Pi-optimized backend with lower RAM usage and manual generation loop
from typing import List, Optional, Protocol, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class KVCacheLike(Protocol):
    """
    Minimal protocol for a quantized KV-cache wrapper.

    Your implementation should store past_key_values (possibly quantized)
    and return dequantized tensors for the next step.
    """

    def store(self, past_key_values: Any) -> None:
        ...

    def get(self) -> Optional[Any]:
        ...


class HFBackendPi:
    """
    Pi-optimized HuggingFace backend with:
    - CPU-only execution
    - optional float16 weights (smaller RAM)
    - manual token-by-token generation loop
      (needed to integrate quantized KV-cache).

    Works with distilgpt2 and other GPT-2 style causal LMs.
    """

    def __init__(
        self,
        model_name: str = "distilgpt2",
        max_ctx: int = 512,
        dtype: str = "float32",  # "float32" (default) or "float16" for smaller RAM
    ) -> None:
        self.model_name = model_name
        self.max_ctx = max_ctx

        # On Raspberry Pi: always CPU
        self.device = "cpu"

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Model
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        if dtype == "float16":
            # Halves model weight memory, but may be slower on CPU.
            self.model = self.model.to(dtype=torch.float16)

        self.model.to(self.device)
        self.model.eval()

        # No gradients anywhere
        torch.set_grad_enabled(False)

    # ------------------------------------------------------------------
    # Token utilities
    # ------------------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def token_count(self, text: str) -> int:
        return len(self.encode(text))

    # ------------------------------------------------------------------
    # Sampling helper (temperature + top-p nucleus)
    # ------------------------------------------------------------------
    @staticmethod
    def _sample_next_token(
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
    ) -> int:
        """
        logits: [vocab_size]
        returns: int token id
        """
        if temperature <= 0:
            # Greedy
            return int(torch.argmax(logits).item())

        # Temperature scaling
        logits = logits / temperature

        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Top-p (nucleus) sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)

        # Keep smallest prefix with cumulative probability >= top_p
        mask = cumulative <= top_p
        # Always keep at least one token
        mask[..., 0] = True

        filtered_probs = sorted_probs * mask
        filtered_probs = filtered_probs / filtered_probs.sum()  # renormalize

        # Sample from the filtered distribution
        next_idx = torch.multinomial(filtered_probs, num_samples=1).item()
        next_token_id = sorted_indices[next_idx].item()
        return int(next_token_id)

    # ------------------------------------------------------------------
    # Custom generation loop with optional quantized KV-cache
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_p: float = 0.95,
        kv_cache: Optional[KVCacheLike] = None,
    ) -> str:
        """
        Manual token-by-token generation, exposing past_key_values.

        If kv_cache is provided, we:
          - store past_key_values after each step (you can quantize there)
          - fetch them from kv_cache before each forward pass
        """

        # Tokenize prompt with context truncation
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_ctx,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        generated: List[int] = []

        # First forward pass: full prompt
        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]          # [1, vocab]
            past_kv = outputs.past_key_values          # tuple of tuples

            if kv_cache is not None:
                kv_cache.store(past_kv)

        # Generate tokens one by one
        for _ in range(max_new_tokens):
            # Optionally retrieve (dequantized) KV-cache
            if kv_cache is not None:
                past_kv = kv_cache.get()
                # If cache is empty for some reason, we could re-run the
                # full prompt, but for now assume it's present.
                if past_kv is None:
                    # Fallback: break or re-run prompt; here we just stop.
                    break

            # logits is [1, vocab]; select token distribution
            next_token_id = self._sample_next_token(
                logits[0], temperature=temperature, top_p=top_p
            )
            generated.append(next_token_id)

            # Prepare next input as a single token
            next_input_ids = torch.tensor(
                [[next_token_id]], device=self.device
            )

            # One-step forward with cached KV
            with torch.inference_mode():
                outputs = self.model(
                    input_ids=next_input_ids,
                    past_key_values=past_kv,
                    use_cache=True,
                )
                logits = outputs.logits[:, -1, :]      # [1, vocab]
                past_kv = outputs.past_key_values

                if kv_cache is not None:
                    kv_cache.store(past_kv)

        return self.decode(generated)
