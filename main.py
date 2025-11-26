#!/usr/bin/env python3
"""
Example driver: query the vector store, call the generator, and save generated tests.

Refactored to initialize Chroma with an embedding model and use the retriever interface.
"""
import os
from dotenv import load_dotenv
from tools import generate_tests, save_files, PERSIST_DIR
from langchain_chroma import Chroma
# Import the embedding factory from ingest.py for consistency
from ingest import create_embeddings
from langchain_core.documents import Document # For type hinting/clarity

load_dotenv()

def run_example():
    # Example user request - replace with your prompt and specifics
    user_request = (
        "Create a pytest file that tests the function `sum` in `my_package/math_utils.py`. "
        "The project prefers simple fixtures and clear assertions. Use style similar to the provided examples."
    )

    # Load embedding model (must be the same one used in ingest.py)
    embeddings = create_embeddings() 
    
    # Load vector DB, explicitly passing embeddings (Refactored L26)
    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    
    # Use as_retriever for idiomatic LangChain retrieval (Refactored L27)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4}) 
    docs: List[Document] = retriever.invoke(user_request)

    # Convert docs into plain strings with filename metadata for style guidance
    context = []
    for d in docs:
        # Simplified access, assuming standard Document structure (Refactored L31-L35)
        meta = d.metadata
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
    # Ensure create_embeddings is imported from ingest to avoid circular imports if possible
    # In a real project, this helper would typically be in its own utility file.
    run_example()