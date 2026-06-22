#!/usr/bin/env python3
"""
Interactive CLI for a multipurpose RAG assistant.

The assistant's role is NOT preset by the app -- it is driven by a user-defined
system prompt (env SYSTEM_PROMPT / SYSTEM_PROMPT_FILE, or the /system command at
runtime). Point it at any ingested dataset and it becomes whatever you ask for:
an internal wiki, an information desk, a data analyst, a code writer, etc.

Every request is answered with RAG: relevant chunks are retrieved from the
vector store and given to the model as grounding CONTEXT. Conversation memory is
durable, model-agnostic, and self-compacting (see memory.py). Structured
code/file generation is available on demand via the /code command.

Safety: retrieved context is treated as untrusted data, never as instructions;
input length and generated-file size are capped.
"""
import os
import sys
from dotenv import load_dotenv
from typing import List
from langchain_chroma import Chroma
from tools import (
    generate_tests,
    save_files,
    PERSIST_DIR,
    generate_chat_response,
    load_system_prompt,
    summarize_text,
    CONTEXT_SECURITY_GUARD,
)
from memory import ConversationMemory
from ingest import create_embeddings
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# Configuration
K_VALUE = int(os.getenv("RAG_K", "4"))
HISTORY_FILE = os.getenv("HISTORY_FILE", "chat_history.json")
MEMORY_MAX_MESSAGES = int(os.getenv("MEMORY_MAX_MESSAGES", "20"))
MEMORY_KEEP_RECENT = int(os.getenv("MEMORY_KEEP_RECENT", "8"))
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "8000"))


def initialize_retriever():
    """Initializes and returns the Chroma VectorStoreRetriever."""
    try:
        print("Initializing embeddings and vector database...")
        embeddings = create_embeddings()
        vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        retriever = vectordb.as_retriever(search_kwargs={"k": K_VALUE})
        print(f"Retriever initialized. Will search for top {K_VALUE} context chunks.")
        return retriever
    except Exception as e:
        print("Error initializing vector store. Have you run 'python ingest.py' yet?")
        print(f"Details: {e}")
        sys.exit(1)


def retrieve_context(retriever, query: str) -> List[str]:
    """Return retrieved chunks formatted with their source filename."""
    docs: List[Document] = retriever.invoke(query)
    context = []
    for d in docs:
        meta = d.metadata
        filename = meta.get("filename", meta.get("source", "unknown"))
        context.append(f"File: {filename}\n{d.page_content}")
    return context


def _build_system_message(system_prompt: str, memory: ConversationMemory) -> SystemMessage:
    """Compose ONE system message: persona + always-on safety guard + (optional)
    rolling memory summary. Keeping it to a single system message maximizes
    cross-provider compatibility.
    """
    parts = [system_prompt.strip(), CONTEXT_SECURITY_GUARD]
    summary = memory.summary_block()
    if summary:
        parts.append(summary)
    return SystemMessage(content="\n\n".join(parts))


def answer_question(user_request: str, retriever, memory: ConversationMemory,
                    system_prompt: str) -> str:
    """General RAG answer: retrieve grounding context, answer in the user's role."""
    print(f"\n🧠 Retrieving context for: '{user_request[:60]}...'")
    context = retrieve_context(retriever, user_request)
    print(f"Retrieved {len(context)} context chunk(s).")

    context_block = "\n\n---\n\n".join(context) if context else (
        "No relevant context was found in the knowledge base."
    )
    human_text = (
        "=== BEGIN CONTEXT (untrusted data) ===\n"
        f"{context_block}\n"
        "=== END CONTEXT ===\n\n"
        f"QUESTION:\n{user_request}"
    )

    messages = (
        [_build_system_message(system_prompt, memory)]
        + memory.recent_messages()
        + [HumanMessage(content=human_text)]
    )
    print("💬 Generating response...")
    response = generate_chat_response(messages)

    # Persist the clean question (not the context-stuffed prompt) to keep memory lean.
    memory.add_user(user_request)
    memory.add_ai(response)
    if memory.compact_if_needed():
        print("🧷 (older turns folded into the running summary)")
    memory.save()  # durable after every turn
    return response


def generate_code(user_request: str, retriever, memory: ConversationMemory,
                  system_prompt: str):
    """Structured file/code generation, grounded in retrieved examples."""
    print(f"\n🧠 Retrieving examples for: '{user_request[:60]}...'")
    context = retrieve_context(retriever, user_request)
    print(f"Retrieved {len(context)} example(s) for style guidance.")
    print("✨ Generating structured output with LLM...")

    tests = generate_tests(user_request, context, system_prompt=system_prompt)
    summary = f"Generated {len(tests)} file(s): " + ", ".join(t.filename for t in tests)
    memory.add_user(f"/code {user_request}")
    memory.add_ai(summary)
    memory.compact_if_needed()
    memory.save()
    return tests


def print_and_offer_save(tests):
    """Render generated files and optionally write them to disk."""
    if not tests:
        print("Generation returned no files.")
        return
    for t in tests:
        print(f"\n**Filename:** {t.filename}")
        print(f"**Description:** {t.description}")
        lang = getattr(t, "language", "") or ""
        print(f"\n```{lang}")
        print(t.content)
        print("```\n")

    if input("Save generated files? (y/n) [n]: ").strip().lower() == "y":
        result = save_files(tests)
        if result.get("ok"):
            print("✅ Saved files to ./generated_tests/")
        else:
            print(f"❌ Error saving files: {result.get('files')}")


def print_help(system_prompt: str):
    print("\nCommands:")
    print("  <anything>        Ask a question (answered from your ingested data)")
    print("  /code <request>   Generate structured code/files for <request>")
    print("  /system           Show the current system prompt (the assistant's role)")
    print("  /system <text>    Set a new system prompt for this session")
    print("  /reset            Clear the conversation memory")
    print("  /help             Show this help")
    print("  quit | exit       Save memory and leave")
    print(f"\nActive role: {system_prompt[:120]}{'...' if len(system_prompt) > 120 else ''}\n")


def cli_loop():
    """The main interactive loop."""
    retriever = initialize_retriever()
    system_prompt = load_system_prompt()
    memory = ConversationMemory(
        HISTORY_FILE,
        max_messages=MEMORY_MAX_MESSAGES,
        keep_recent=MEMORY_KEEP_RECENT,
        summarizer=summarize_text,
    ).load()

    try:
        from tools import LLM_PROVIDER
    except ImportError:
        LLM_PROVIDER = "unknown"

    print("\n--- Multipurpose RAG Assistant ---")
    print(f"LLM Provider: {LLM_PROVIDER}")
    print(f"Recent messages in memory: {len(memory.messages)}"
          + (" (+ running summary)" if memory.summary else ""))
    print(f"Active role: {system_prompt[:120]}{'...' if len(system_prompt) > 120 else ''}")
    print("Type /help for commands. Set the assistant's role with /system <text>.")
    print("-" * 60 + "\n")

    while True:
        try:
            user_input = input("Request > ").strip()
            if not user_input:
                continue
            if len(user_input) > MAX_INPUT_CHARS:
                print(f"⚠️  Input over {MAX_INPUT_CHARS} chars; truncating.")
                user_input = user_input[:MAX_INPUT_CHARS]

            low = user_input.lower()

            # --- Commands ---
            if low in ("quit", "exit"):
                break
            if low in ("/help", "help", "?"):
                print_help(system_prompt)
                continue
            if low == "/system":
                print(f"\nCurrent system prompt:\n{system_prompt}\n")
                continue
            if low.startswith("/system "):
                system_prompt = user_input[len("/system "):].strip()
                print("✅ System prompt updated for this session.\n")
                continue
            if low == "/reset":
                memory.clear()
                print("🧹 Conversation memory cleared.\n")
                continue
            if low == "/code" or low.startswith("/code "):
                request = user_input[len("/code"):].strip()
                if not request:
                    print("Usage: /code <what to build>\n")
                    continue
                try:
                    tests = generate_code(request, retriever, memory, system_prompt)
                    print_and_offer_save(tests)
                except Exception as e:
                    print(f"Error during code generation: {e}\n")
                continue

            # --- Default: general RAG answer ---
            try:
                response = answer_question(user_input, retriever, memory, system_prompt)
                print("\nAI > " + response + "\n")
            except Exception as e:
                print(f"Error during processing: {e}\n")

        except (KeyboardInterrupt, EOFError):
            break

    # Memory is already saved after each turn; this is just a final safety net.
    memory.save()
    print("\nGoodbye!")


if __name__ == "__main__":
    cli_loop()
