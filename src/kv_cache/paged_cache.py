class PagedCache:
    """
    Ring-buffer over message turns. Keeps last N turns while preserving role tags.
    """
    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self.turns = []

    def apply(self, backend, history_text: str, new_user: str) -> str:
        self.turns.append(("User", new_user))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]
        convo = []
        for role, text in self.turns:
            convo.append(f"{role}: {text}")
            if role == "User":
                convo.append("Assistant:")
        joined = "\n".join(convo)
        # Light token trimming as guard
        while backend.tokenize_len(joined) > 512:
            self.turns = self.turns[1:]
            joined = "\n".join([f"{r}: {t}" if r=="User" else "Assistant:" for r,t in self.turns])
        return joined
