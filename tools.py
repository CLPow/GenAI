#!/usr/bin/env python3
"""
Tooling: retrieval + generation + save.

This version uses ChatGoogleGenerativeAI when LLM_PROVIDER=google (and langchain_google_genai installed).
Fallback: uses ChatOpenAI if env indicates OpenAI or if import fails.
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel

load_dotenv()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google").lower()

# LangChain imports
from langchain.vectorstores import Chroma
from langchain.schema import SystemMessage, HumanMessage

# Chat model imports: try Google first
ChatGoogleGenerativeAI = None
try:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
    ChatGoogleGenerativeAI = ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None

# Fallback to ChatOpenAI if available
ChatOpenAI = None
try:
    from langchain.chat_models import ChatOpenAI  # type: ignore
    ChatOpenAI = ChatOpenAI
except Exception:
    ChatOpenAI = None

# Document model for generated tests
class GeneratedTest(BaseModel):
    filename: str
    language: str
    framework: str
    content: str
    description: str
    metadata: dict | None = None


PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")


def get_retriever(persist_dir: str = PERSIST_DIR, k: int = 4):
    if not Path(persist_dir).exists():
        raise FileNotFoundError(f"Chroma DB not found in {persist_dir}; run ingest.py first")
    vectordb = Chroma(persist_directory=persist_dir)
    return vectordb.as_retriever(search_type="similarity", search_kwargs={"k": k})


def _make_chat_model(temperature: float = 0.0):
    """
    Factory that returns a chat model instance. If provider==google and langchain_google_genai installed,
    returns a ChatGoogleGenerativeAI instance. Otherwise falls back to ChatOpenAI.
    """
    if LLM_PROVIDER == "google" and ChatGoogleGenerativeAI is not None:
        # instantiate Gemini / Google Generative model
        # model name can be "gemini-2.5-flash" or another available one in your account
        try:
            return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temperature)
        except TypeError:
            # Some wrappers accept different kwargs — fall back to default constructor
            return ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    if ChatOpenAI is not None:
        # Use OpenAI Chat model as fallback
        return ChatOpenAI(temperature=temperature, model_name="gpt-4o-mini")
    raise RuntimeError("No usable chat model found. Install langchain_google_genai or set up ChatOpenAI.")


def generate_tests(prompt_request: str, retrieved_context: List[str], temperature: float = 0.0) -> List[GeneratedTest]:
    """
    Use an LLM to generate 1..N test files as JSON array. This function instructs the model to return JSON only.
    """
    chat = _make_chat_model(temperature=temperature)

    system_text = (
        "You are an assistant that writes runnable test scripts. "
        "Given style examples and a user request, return a JSON array of objects with keys: "
        "filename, language, framework, content, description, metadata. "
        "Return ONLY valid JSON (no extra commentary)."
    )
    system_msg = SystemMessage(content=system_text)

    examples_text = "\n\n---\n\n".join(retrieved_context) if retrieved_context else "No examples provided."

    human_text = (
        f"CONTEXT EXAMPLES (style hints):\n{examples_text}\n\n"
        f"USER REQUEST:\n{prompt_request}\n\n"
        "Return only a JSON array. Generate 1..3 tests as needed. Keep code runnable and include imports and run instructions."
    )
    human_msg = HumanMessage(content=human_text)

    # Call model. LangChain chat models accept a list of messages.
    # ChatGoogleGenerativeAI and ChatOpenAI both accept calling with [SystemMessage, HumanMessage]
    resp = chat([system_msg, human_msg])
    # The returned object shape may vary by langchain version; attempt safe extraction:
    raw = ""
    try:
        # common interface: resp.content or resp[0].content
        raw = getattr(resp, "content", None) or (resp[0].content if isinstance(resp, (list, tuple)) else None) or str(resp)
    except Exception:
        raw = str(resp)

    raw = raw.strip()
    # Parse JSON. Be forgiving if the model included backticks or code fences.
    try:
        parsed = json.loads(raw)
    except Exception:
        import re
        m = re.search(r"(\[.*\])", raw, re.S)
        if m:
            parsed = json.loads(m.group(1))
        else:
            raise ValueError(f"Failed to parse JSON from model output.\nModel output:\n{raw}")

    tests = []
    for obj in parsed:
        tests.append(GeneratedTest(**obj))
    return tests


def save_files(tests: List[GeneratedTest], out_dir: str = "./generated_tests") -> dict:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    results = []
    for t in tests:
        filename = t.filename
        target = Path(out_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(target, "w", encoding="utf-8") as fh:
                fh.write(t.content)
            results.append({"filename": str(target), "ok": True})
        except Exception as e:
            results.append({"filename": str(target), "ok": False, "error": str(e)})
    return {"ok": all(r.get("ok") for r in results), "files": results}