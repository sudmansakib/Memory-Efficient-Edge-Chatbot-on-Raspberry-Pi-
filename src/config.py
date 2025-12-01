from pydantic import BaseModel
class Settings(BaseModel):
    log_dir: str = "data/logs"
    max_ctx: int = 512
    cache_type: str = "sliding"   # sliding | paged | quantized
    backend: str = "llama"        # llama | hf
    model_path: str = "models/tinyllama-q4_0.gguf"
settings = Settings()
