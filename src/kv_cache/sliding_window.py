# src/kv_cache/sliding_window.py

from typing import List, Tuple, Protocol


class BackendLike(Protocol):
    def encode(self, text: str) -> List[int]:
        ...

    def decode(self, token_ids: List[int]) -> str:
        ...

    def token_count(self, text: str) -> int:
        ...


class SlidingWindowCache:
    """
    Sliding-window conversation cache.
    Keeps only the most recent tokens (max_tokens).
    """

    def __init__(self, max_tokens: int = 512) -> None:
        self.max_tokens = max_tokens
        # list of (user, assistant) turns
        self.turns: List[Tuple[str, str]] = []

    def add_turn(self, user: str, assistant: str) -> None:
        self.turns.append((user, assistant))

    def _render_prompt(self, new_user: str) -> str:
        lines: List[str] = []
        for u, a in self.turns:
            lines.append(f"User: {u}")
            lines.append(f"Assistant: {a}")
        lines.append(f"User: {new_user}")
        lines.append("Assistant:")
        return "\n".join(lines)

    def build_prompt(self, backend: BackendLike, new_user: str) -> str:
        """
        Returns a prompt string with prior context plus new_user.
        Drops oldest turns until token budget fits.
        """
        if self.max_tokens <= 0:
            return f"User: {new_user}\nAssistant:"

        # Start from full history
        prompt = self._render_prompt(new_user)

        # While too long, drop oldest turn
        while backend.token_count(prompt) > self.max_tokens and self.turns:
            self.turns.pop(0)
            prompt = self._render_prompt(new_user)

        # If still too long (single huge user message), hard truncate by tokens
        if backend.token_count(prompt) > self.max_tokens:
            token_ids = backend.encode(prompt)
            token_ids = token_ids[-self.max_tokens :]
            prompt = backend.decode(token_ids)

        return prompt
