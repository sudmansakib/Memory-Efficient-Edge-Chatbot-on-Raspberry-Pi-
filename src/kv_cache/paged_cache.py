# src/kv_cache/paged_cache.py

from typing import List, Tuple, Protocol


class BackendLike(Protocol):
    def token_count(self, text: str) -> int:
        ...


class PagedCache:
    """
    Paged / rotating cache.
    Keeps a fixed number of most recent turns (max_turns).
    Has an optional token budget as a safety cap.
    """

    def __init__(self, max_turns: int = 6, max_tokens: int | None = None) -> None:
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.turns: List[Tuple[str, str]] = []

    def add_turn(self, user: str, assistant: str) -> None:
        self.turns.append((user, assistant))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def _render_prompt(self, new_user: str) -> str:
        lines: List[str] = []
        for u, a in self.turns:
            lines.append(f"User: {u}")
            lines.append(f"Assistant: {a}")
        lines.append(f"User: {new_user}")
        lines.append("Assistant:")
        return "\n".join(lines)

    def build_prompt(self, backend: BackendLike, new_user: str) -> str:
        prompt = self._render_prompt(new_user)

        if self.max_tokens is None:
            return prompt

        # If token budget given, drop turns until we fit
        while backend.token_count(prompt) > self.max_tokens and self.turns:
            self.turns.pop(0)
            prompt = self._render_prompt(new_user)

        return prompt
