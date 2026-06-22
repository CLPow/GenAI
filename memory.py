#!/usr/bin/env python3
"""
Durable, model-agnostic conversation memory with rolling-summary compaction.

Why this exists:
- Reliability: writes are atomic (temp file + os.replace) and happen after every
  turn, so a crash or hard-close never corrupts or loses memory.
- Model independence: memory is persisted as a natural-language summary plus
  plain message dicts -- nothing provider-specific. Switch Gemini / OpenAI /
  Ollama and the same file keeps the assistant's recollection consistent. A
  single shared file therefore gives consistent behavior across models.
- Bounded context: when the transcript grows past a threshold, the oldest turns
  are folded into the summary (via an injected summarizer) and dropped, keeping
  prompts small and stable regardless of how long the session runs.

This module depends only on the standard library and langchain_core message
types, so it imports cleanly without pulling in any LLM SDK (the summarizer is
injected by the caller to avoid circular imports).
"""
import json
import os
import tempfile
from pathlib import Path
from typing import Callable, List, Optional

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    messages_from_dict,
    messages_to_dict,
)

# A summarizer takes a block of text and returns a condensed summary string.
SummarizerFn = Callable[[str], str]

_ROLE_LABELS = {"human": "User", "ai": "Assistant", "system": "System"}


def _normalize(message: BaseMessage) -> Optional[BaseMessage]:
    """Coerce a loaded message to a clean string-content form, or drop it.

    Different providers may have stored content as a list of blocks; flatten to
    text so the history is portable across models. Empty messages are dropped.
    """
    content = getattr(message, "content", "")
    if isinstance(content, list):
        content = " ".join(str(part) for part in content)
    content = str(content).strip()
    if not content:
        return None
    message.content = content
    return message


def _render(messages: List[BaseMessage]) -> str:
    """Render messages as a plain User/Assistant transcript for summarization."""
    lines = []
    for m in messages:
        label = _ROLE_LABELS.get(getattr(m, "type", ""), getattr(m, "type", "?"))
        lines.append(f"{label}: {m.content}")
    return "\n".join(lines)


class ConversationMemory:
    """Crash-resistant, self-compacting conversation memory."""

    # Refuse to load absurdly large / tampered files into memory.
    MAX_FILE_BYTES = 5 * 1024 * 1024

    def __init__(self, path: str, *, max_messages: int = 20, keep_recent: int = 8,
                 summarizer: Optional[SummarizerFn] = None):
        self.path = Path(path)
        self.max_messages = max(2, max_messages)
        self.keep_recent = max(2, min(keep_recent, self.max_messages))
        self.summarizer = summarizer
        self.summary: str = ""
        self.messages: List[BaseMessage] = []

    # --- persistence ---------------------------------------------------------
    def load(self) -> "ConversationMemory":
        if not self.path.exists():
            return self
        try:
            if self.path.stat().st_size > self.MAX_FILE_BYTES:
                print(f"Warning: {self.path} exceeds {self.MAX_FILE_BYTES} bytes; starting with empty memory.")
                return self
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                # Backward compatibility with the old bare-list history format.
                self.summary, raw = "", data
            else:
                self.summary = str(data.get("summary", "") or "")
                raw = data.get("messages", [])
            loaded = messages_from_dict(raw) if raw else []
            self.messages = [m for m in (_normalize(m) for m in loaded) if m is not None]
        except Exception as e:
            print(f"Warning: failed to load memory ({e}). Starting fresh.")
            self.summary, self.messages = "", []
        return self

    def save(self) -> None:
        """Persist atomically: write a temp file in the same dir, then replace."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"summary": self.summary, "messages": messages_to_dict(self.messages)}
            fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False)
                os.replace(tmp, self.path)  # atomic on POSIX and Windows
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
        except Exception as e:
            print(f"Error: failed to save memory: {e}")

    # --- mutation ------------------------------------------------------------
    def add_user(self, text: str) -> None:
        self.messages.append(HumanMessage(content=str(text)))

    def add_ai(self, text: str) -> None:
        self.messages.append(AIMessage(content=str(text)))

    def clear(self) -> None:
        self.summary, self.messages = "", []
        self.save()

    # --- retrieval for the LLM ----------------------------------------------
    def recent_messages(self) -> List[BaseMessage]:
        """Verbatim recent turns (history minus what's been summarized away)."""
        return list(self.messages)

    def summary_block(self) -> str:
        """A text block describing earlier conversation, or '' if none.

        Returned as plain text so the caller can fold it into the single system
        message -- avoiding multiple system messages, which some providers reject.
        """
        if not self.summary:
            return ""
        return f"Summary of earlier conversation (for continuity):\n{self.summary}"

    # --- compaction ----------------------------------------------------------
    def compact_if_needed(self) -> bool:
        """Fold old turns into the rolling summary once the transcript is too long.

        Returns True if compaction ran. Without a summarizer it still bounds
        memory by trimming the oldest turns (stable, just less context-aware).
        """
        if len(self.messages) <= self.max_messages:
            return False

        n_old = len(self.messages) - self.keep_recent
        old, recent = self.messages[:n_old], self.messages[-self.keep_recent:]

        if not self.summarizer:
            self.messages = recent
            return True

        prior = f"PREVIOUS SUMMARY:\n{self.summary}\n\n" if self.summary else ""
        summary_input = prior + "NEW MESSAGES TO FOLD IN:\n" + _render(old)
        try:
            new_summary = (self.summarizer(summary_input) or "").strip()
            if new_summary:
                self.summary = new_summary
                self.messages = recent
                return True
        except Exception as e:
            print(f"Warning: memory compaction failed ({e}); trimming oldest turns instead.")
        # Fallback: keep memory bounded even if summarization failed.
        self.messages = recent
        return True
