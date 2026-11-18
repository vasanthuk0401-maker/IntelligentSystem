from __future__ import annotations

from typing import Callable, List, Optional

from core.llm import LLM, get_llm


class BaseAgent:
    name = "Agent"

    def __init__(self, llm: Optional[LLM] = None, trace: Optional[List[str]] = None,
                 on_log: Optional[Callable[[str], None]] = None):
        self.llm = llm or get_llm()
        self.trace = trace if trace is not None else []
        self.on_log = on_log

    def log(self, msg: str) -> None:
        line = f"[{self.name}] {msg}"
        self.trace.append(line)
        if self.on_log:
            self.on_log(line)
