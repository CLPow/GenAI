#!/usr/bin/env python3
"""
Tooling: LLM provider factory + structured test generation + file saving.

Design goals:
- Balanced cloud / local support. One env var, `LLM_PROVIDER`, picks the backend:
  google | openai (cloud) or ollama (local, zero keys required).
- Strict selection: the chosen provider is the one that runs. No silent fallback
  to a *different* cloud provider (that hides cost + leaks intent).
- Lazy, per-provider imports: a missing cloud SDK never breaks the local path
  (and vice versa).
- No secret values are printed or logged.
"""
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

load_dotenv()

# --- Configuration (all overridable via .env) ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google").strip().lower()
LLM_TEMP = float(os.getenv("LLM_TEMPERATURE", "0.0"))
GOOGLE_MODEL_NAME = os.getenv("GOOGLE_MODEL_NAME", "gemini-2.5-flash")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "gemma3:1b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

SUPPORTED_PROVIDERS = ("google", "openai", "ollama")
CLOUD_PROVIDERS = ("google", "openai")

print(f"LLM Provider: {LLM_PROVIDER}")


# --- Document models for generated tests ---
class GeneratedTest(BaseModel):
    filename: str = Field(description="Suggested filename for the test script (e.g., test_math.py).")
    language: str = Field(description="Primary programming language (e.g., python).")
    framework: str = Field(description="Testing framework used (e.g., pytest, unittest, jest).")
    content: str = Field(description="Full, runnable test code, including imports.")
    description: str = Field(description="A brief description of what this test does.")
    metadata: Optional[dict] = Field(description="Optional extra metadata.", default=None)


class GeneratedTestList(BaseModel):
    tests: List[GeneratedTest] = Field(description="A JSON array of 1 to 3 GeneratedTest objects.")


def _require_key(var_name: str, provider: str):
    """Fail fast with a clear, secret-free message when a cloud key is missing."""
    if not os.getenv(var_name):
        raise RuntimeError(
            f"LLM_PROVIDER='{provider}' requires {var_name} to be set in your environment/.env. "
            f"Set it, or switch to a local provider with LLM_PROVIDER=ollama (no key needed)."
        )


def _make_chat_model(temperature: float = LLM_TEMP):
    """Return a chat-model instance for the configured provider.

    Each provider is imported lazily inside its own branch so that a missing
    SDK for one backend never breaks the others. The chosen provider is the
    only one attempted -- no cross-provider fallback.
    """
    if LLM_PROVIDER not in SUPPORTED_PROVIDERS:
        raise RuntimeError(
            f"Unsupported LLM_PROVIDER='{LLM_PROVIDER}'. "
            f"Choose one of: {', '.join(SUPPORTED_PROVIDERS)}."
        )

    # --- LOCAL: Ollama (no API key required) ---
    if LLM_PROVIDER == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as e:
            raise RuntimeError(
                "LLM_PROVIDER='ollama' but langchain-ollama is not installed. "
                "Run: pip install langchain-ollama"
            ) from e
        return ChatOllama(
            model=OLLAMA_MODEL_NAME,
            base_url=OLLAMA_BASE_URL,
            temperature=temperature,
        )

    # --- CLOUD: Google Gemini ---
    if LLM_PROVIDER == "google":
        _require_key("GOOGLE_API_KEY", "google")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise RuntimeError(
                "LLM_PROVIDER='google' but langchain-google-genai is not installed. "
                "Run: pip install langchain-google-genai"
            ) from e
        return ChatGoogleGenerativeAI(model=GOOGLE_MODEL_NAME, temperature=temperature)

    # --- CLOUD: OpenAI ---
    if LLM_PROVIDER == "openai":
        _require_key("OPENAI_API_KEY", "openai")
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise RuntimeError(
                "LLM_PROVIDER='openai' but langchain-openai is not installed. "
                "Run: pip install langchain-openai"
            ) from e
        return ChatOpenAI(model=OPENAI_MODEL_NAME, temperature=temperature)

    raise RuntimeError("No usable chat model found.")  # pragma: no cover


def generate_chat_response(messages: List[BaseMessage], temperature: float = 0.7) -> str:
    """Generate a free-form text response.

    All providers are returned as chat models, so the interface is uniform:
    `.invoke(messages)` -> message with `.content`. A defensive getattr keeps
    this working even if a backend returns a raw string.
    """
    chat = _make_chat_model(temperature=temperature)
    try:
        response = chat.invoke(messages)
        return getattr(response, "content", response)
    except Exception as e:
        raise RuntimeError(f"Failed to generate chat response from model: {e}") from e


def generate_tests(prompt_request: str, retrieved_context: list, temperature: float = LLM_TEMP):
    """Generate structured test code via the structured-output chain."""
    chat = _make_chat_model(temperature=temperature)
    chat_with_structured_output = chat.with_structured_output(GeneratedTestList)

    system_text = (
        "You are an assistant that writes runnable test scripts. "
        "Your task is to return a JSON object that strictly adheres to the provided schema. "
        "The 'tests' key must contain an array of 1..3 test objects. "
        "Ensure the 'content' is runnable code including all necessary imports."
    )

    examples_text = "\n\n---\n\n".join(retrieved_context) if retrieved_context else "No examples provided."
    human_text = (
        f"CONTEXT EXAMPLES (style hints):\n{examples_text}\n\n"
        f"USER REQUEST:\n{prompt_request}\n\n"
        "Return only a JSON object that matches the schema. Generate 1 to 3 tests as needed."
    )

    messages = [SystemMessage(content=system_text), HumanMessage(content=human_text)]

    try:
        parsed_response: GeneratedTestList = chat_with_structured_output.invoke(messages)
        return parsed_response.tests
    except Exception as e:
        raise RuntimeError(f"Failed to generate structured response from model: {e}") from e


# Filenames we are willing to write from model output.
_ALLOWED_SUFFIXES = {".py", ".js", ".ts", ".java", ".cpp", ".go", ".rb", ".php", ".rs", ".txt"}


def save_files(tests: List[GeneratedTest], out_dir: str = "./generated_tests") -> dict:
    """Write generated tests to disk, sandboxed inside out_dir.

    Hardening: strip any path components from the model-supplied filename and
    reject unexpected extensions, so a malicious/garbled filename can't escape
    the output directory or drop an executable elsewhere.
    """
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    results = []

    for t in tests:
        safe_filename = os.path.basename(t.filename or "").strip()
        suffix = Path(safe_filename).suffix.lower()

        if not safe_filename or suffix not in _ALLOWED_SUFFIXES:
            results.append({
                "filename": t.filename,
                "ok": False,
                "error": f"Rejected filename (empty or disallowed extension '{suffix}').",
            })
            continue

        target = (out_path / safe_filename).resolve()
        # Final guard: target must stay within out_path.
        if out_path != target.parent:
            results.append({"filename": safe_filename, "ok": False, "error": "Path escape blocked."})
            continue

        try:
            with open(target, "w", encoding="utf-8") as fh:
                fh.write(t.content)
            results.append({"filename": str(target), "ok": True})
        except Exception as e:
            results.append({"filename": str(target), "ok": False, "error": str(e)})

    return {"ok": bool(results) and all(r.get("ok") for r in results), "files": results}
