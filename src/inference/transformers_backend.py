import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class HFBackend:
    def __init__(self, model_name="distilgpt2", max_ctx=512):
        self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.max_ctx = max_ctx

    def generate(self, prompt: str, max_tokens: int = 128):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.8)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):]

    def tokenize_len(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
