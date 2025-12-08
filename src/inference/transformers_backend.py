# src/inference/transformers_backend.py

from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class HFBackend:
    """
    Simple HuggingFace backend (distilgpt2 by default).
    CPU-only; good enough for development and Pi deployment.
    """

    def __init__(
        self,
        model_name: str = "distilgpt2",
        max_ctx: int = 512,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_ctx = max_ctx

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    # --- token utilities ---

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def token_count(self, text: str) -> int:
        return len(self.encode(text))

    # --- generation ---

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> str:

        # Build tokenizer inputs (this includes attention_mask)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Run generation
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,   # Fixes HF warning
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Extract only the newly-generated portion
        gen_ids = output_ids[0, input_ids.shape[-1]:]
        return self.decode(gen_ids)