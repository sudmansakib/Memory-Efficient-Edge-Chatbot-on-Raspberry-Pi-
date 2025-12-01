class SlidingWindowCache:
    def __init__(self, max_ctx: int):
        self.max_ctx = max_ctx

    def apply(self, backend, history_text: str, new_user: str) -> str:
        combined = history_text + f"\nUser: {new_user}\nAssistant:"
        # Trim by token count
        while backend.tokenize_len(combined) > self.max_ctx:
            # Drop oldest line (very cheap + effective on Pi)
            parts = combined.split("\n", 2)
            combined = parts[2] if len(parts) > 2 else combined[-2000:]
        return combined
