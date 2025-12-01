from llama_cpp import Llama

class LlamaBackend:
    def __init__(self, model_path: str, max_ctx: int):
        self.llm = Llama(model_path=model_path, n_ctx=max_ctx, n_gpu_layers=0)

    def generate(self, prompt: str, max_tokens: int = 128):
        out = self.llm(prompt, max_tokens=max_tokens, stream=False)
        return out["choices"][0]["text"]

    def tokenize_len(self, text: str) -> int:
        return len(self.llm.tokenize(text.encode()))
