# Test-Script Generator (RAG) — Starter Project

Overview
- This project ingests plain `.txt` test scripts and related docs, indexes them into a vector store, and exposes a simple agent pipeline that retrieves relevant examples and asks an LLM to generate new test scripts in the same style.

Folder layout
- data/                # drop your .txt test scripts here
- ingest.py            # splits files into chunks and persists into a vector DB (Chroma by default)
- tools.py             # tool wrappers: generate_tests_tool, save_tool, retrieval helpers
- main.py              # example usage: query retriever + LLM to generate tests
- requirements.txt
- .env.sample

Quickstart
1. Create a Python 3.10+ virtualenv and install requirements:
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2. Put your .txt test scripts in the `data/` directory. Recommend file metadata in filenames, e.g. `pytest_my_module_auth_flow.txt`, but metadata can also be added later.

3. Copy `.env.sample` to `.env` and set your API keys (e.g., OPENAI_API_KEY, or configure Google/Gemini per your provider).

4. Ingest files:
   python ingest.py --data-dir ./data --persist-dir ./chroma_db

   This creates/updates a local Chroma DB.

5. Run generator:
   python main.py

   The script loads retriever context and sends a generation request to the LLM. The generated test(s) will be printed and saved (by default, into ./generated_tests/).

Important notes
- Security: Do not upload secrets. Sanitize any credentials inside files before ingestion.
- Execution: Generated tests may contain arbitrary code. Only execute them in a sandboxed environment.
- Embeddings: By default, the code uses OpenAI embeddings. You can swap in any embeddings provider supported by LangChain or Chroma.
- Tuning: Use k=3..6 retrieved examples for best results; keep the LLM temperature low for deterministic code output.

If you want, I can:
- Adapt this to Gemini embeddings & chat generation specifically.
- Add a runner tool that executes generated tests inside a Docker container and returns results (careful with security).
- Add a web UI for uploading files and requesting generations.
