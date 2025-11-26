#!/usr/bin/env python3
"""
Tooling: generation + save. Retriever handled via similarity_search in main.py.
"""
import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel as V1BaseModel
from pydantic import Field

load_dotenv()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google").lower()
LLM_TEMP = float(os.getenv("LLM_TEMPERATURE", 0.0))
GOOGLE_MODEL_NAME = os.getenv("GOOGLE_MODEL_NAME", "gemini-2.5-flash")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini") 

print(f"LLM Provider: {LLM_PROVIDER}")
print(f"GOOGLE_API_KEY is loaded: {bool(os.getenv('GOOGLE_API_KEY'))}")

# Chat model imports
ChatGoogleGenerativeAI = None
try:
    # Ensure this core package is installed: pip install langchain-google-genai
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
    from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
    # The problematic langchain_core.pydantic_v1 import is removed here.
    from langchain_core.output_parsers import PydanticOutputParser
    # Note: We rely on 'BaseModel as V1BaseModel' from the top of the file.
except Exception as e:
    # Keep the prints for final diagnostic confirmation
    print("🛑 IMPORT FAILED: ChatGoogleGenerativeAI or a dependency is missing/conflicting.")
    print(f"🛑 Import Error Details: {e}")
    ChatGoogleGenerativeAI = None

ChatOpenAI = None
try:
    from langchain.chat_models import ChatOpenAI  # type: ignore
    from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
    from langchain_core.pydantic_v1 import BaseModel as V1BaseModel
    from langchain_core.output_parsers import PydanticOutputParser
except Exception:
    ChatOpenAI = None


# Document model for generated tests
class GeneratedTest(V1BaseModel):
    filename: str = Field(description="The suggested filename for the test script (e.g., test_math.py).")
    language: str = Field(description="The primary programming language of the test (e.g., python).")
    framework: str = Field(description="The testing framework used (e.g., pytest, unittest, jest).")
    content: str = Field(description="The full, runnable code content of the test script, including imports.")
    description: str = Field(description="A brief description of what this specific test does.")
    metadata: dict | None = Field(description="Optional extra metadata.", default=None)

# Define the expected full output structure (a list of tests)
class GeneratedTestList(V1BaseModel):
    tests: List[GeneratedTest] = Field(description="A JSON array containing 1 to 3 GeneratedTest objects.")

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")


def _make_chat_model(temperature: float = LLM_TEMP):
    """Factory that returns a chat model instance."""
    if LLM_PROVIDER == "google" and ChatGoogleGenerativeAI:
        try:
            return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temperature)
        except Exception as e:
            # THIS IS THE CRITICAL CHANGE: Print the actual error
            print(f"🛑 CRITICAL ERROR: Failed to initialize ChatGoogleGenerativeAI.")
            print(f"🛑 Reason: {e}")
            pass
    if ChatOpenAI:
        return ChatOpenAI(temperature=temperature, model_name=OPENAI_MODEL_NAME)
    raise RuntimeError("No usable chat model found.")


def generate_chat_response(messages: List[BaseMessage], temperature: float = 0.7) -> str:
    """
    Generates a free-form text response using the chat model and conversation history.
    Uses a higher temperature for conversational flow.
    """
    chat = _make_chat_model(temperature=temperature)
    
    try:
        # Invoke the chat model directly with the full message history
        response = chat.invoke(messages)
        return response.content
        
    except Exception as e:
        raise RuntimeError(f"Failed to generate chat response from model: {e}")


def generate_tests(prompt_request: str, retrieved_context: list, temperature: float = LLM_TEMP):
    """Generates structured test code using the structured output chain."""
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

    messages = [
        SystemMessage(content=system_text),
        HumanMessage(content=human_text)
    ]

    try:
        parsed_response: GeneratedTestList = chat_with_structured_output.invoke(messages)
        return parsed_response.tests 
        
    except Exception as e:
        raise RuntimeError(f"Failed to generate structured response from model: {e}")


def save_files(tests: List[GeneratedTest], out_dir: str = "./generated_tests") -> dict:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    results = []
    for t in tests:
        target = Path(out_dir) / t.filename
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(target, "w", encoding="utf-8") as fh:
                fh.write(t.content)
            results.append({"filename": str(target), "ok": True})
        except Exception as e:
            results.append({"filename": str(target), "ok": False, "error": str(e)})
    return {"ok": all(r.get("ok") for r in results), "files": results}