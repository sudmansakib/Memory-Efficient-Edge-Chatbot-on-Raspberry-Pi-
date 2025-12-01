from fastapi import FastAPI
from pydantic import BaseModel
from src.utils.logger import Logger
from src.inference.llama_backend import LlamaBackend
from src.kv_cache.sliding_window import SlidingWindowCache

app = FastAPI()
logger = Logger()
backend = LlamaBackend("models/tinyllama-q4_0.gguf", max_ctx=512)
cache = SlidingWindowCache(max_ctx=512)
history = "You are a helpful assistant."

class Msg(BaseModel):
    text: str

@app.post("/chat")
def chat(m: Msg):
    global history
    prompt = cache.apply(backend, history, m.text)
    out = backend.generate(prompt, max_tokens=128)
    history = prompt + out
    logger.log_interaction("user", m.text)
    logger.log_interaction("assistant", out)
    logger.log_metrics({})
    return {"reply": out.strip()}
