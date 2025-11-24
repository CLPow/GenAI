#!/usr/bin/env python3
"""
Example driver: query the retriever, call the generator, and save generated tests.

Usage:
    python main.py
"""
import os
from dotenv import load_dotenv
from tools import get_retriever, generate_tests, save_files

load_dotenv()

def run_example():
    # Example user request - replace with your prompt and specifics
    user_request = (
        "Create a pytest file that tests the function `sum` in `my_package/math_utils.py`. "
        "The project prefers simple fixtures and clear assertions. Use style similar to the provided examples."
    )

    # Load retriever and fetch top k context chunks
    retriever = get_retriever()
    docs = retriever.get_relevant_documents(user_request)  # returns Document objects
    # Convert docs into plain strings with filename metadata for style guidance
    context = []
    for d in docs:
        meta = d.metadata if hasattr(d, "metadata") else {}
        filename = meta.get("filename", meta.get("source", "unknown"))
        piece = f"File: {filename}\n{d.page_content}"
        context.append(piece)

    print("Retrieved examples (count):", len(context))

    # Generate tests using the LLM guided by retrieved context
    tests = generate_tests(user_request, context, temperature=0.0)
    print(f"Model generated {len(tests)} test(s).")

    # Print a quick preview and save the tests
    for t in tests:
        print("------")
        print("Filename:", t.filename)
        print("Description:", t.description)
        print(t.content[:1000])  # preview top 1000 chars
        print("------")

    save_result = save_files(tests)
    print("Save result:", save_result)

if __name__ == "__main__":
    run_example()