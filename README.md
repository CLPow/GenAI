# 📄 Project Readme: RAG Code Generator AI

🎯 Overview
This project is an Retrieval-Augmented Generation (RAG) system designed to automatically generate high-quality unit tests based on your existing codebase style and specific user requests. It uses a Chroma vector database for retrieval and supports a variety of Large Language Models (LLMs) for generation, including Google Gemini, OpenAI, and local models via Ollama.

Key Features:

Context-Aware Generation: Uses RAG to retrieve examples from your codebase's style guide and existing tests.

Structured Output: Generates tests as clean, parsed JSON objects defined by Pydantic models.

Flexible Deployment: Easily switch between cloud APIs and local LLMs.
__________________________________________________________________________________________________________________

⚙️ Setup and Installation
This project requires Python 3.11 or higher due to modern dependency requirements (LangChain V0.2+ and Pydantic V2).

1. Clone the Repository and Set Up the Environment
```
# Clone the repository
git clone https://github.com/CLPow/GenAI.git
cd your-repo-name

# Create a new virtual environment (Requires Python 3.11+ to be installed)
# Use 'py -3.11 -m venv venv' if 'python' defaults to an older version.
python -m venv venv

# Activate the environment
# Windows (PowerShell):
.\venv\Scripts\Activate
# Linux/macOS:
# source venv/bin/activate
```
2. Install Dependencies
Install the core packages (always required):
```
pip install -U python-dotenv pydantic langchain-core langchain-community langchain-chroma langchain-text-splitters langchain-huggingface sentence-transformers
```

Then install ONLY the provider packages you intend to use. SDKs are imported
lazily per-provider, so a missing one never breaks the others:
```
# Google Gemini (cloud)
pip install -U langchain-google-genai

# OpenAI (cloud)
pip install -U langchain-openai

# Ollama (local, no API key required)
pip install -U langchain-ollama
```

3. Configure Provider and Keys
Create a file named .env in the root directory. `LLM_PROVIDER` is the single
switch that decides which backend runs. The chosen provider is the only one
attempted — there is no silent fallback to a different cloud provider. The same
provider is used for both generation and the RAG embeddings, so keep it
consistent between `ingest.py` and `cli_chat.py` runs (changing the embedding
model after ingest will break vector lookups).
```
# --- LLM Provider Selection ---
# google | openai (cloud)  |  ollama (local, no key)
LLM_PROVIDER=google

# Optional global generation temperature (default 0.0)
# LLM_TEMPERATURE=0.0

# --- Google Gemini Configuration (LLM_PROVIDER=google) ---
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY_HERE
# GOOGLE_MODEL_NAME=gemini-2.5-flash   # optional override

# --- OpenAI Configuration (LLM_PROVIDER=openai) ---
# OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE
# OPENAI_MODEL_NAME=gpt-4o-mini        # optional override

# --- Ollama / Local Configuration (LLM_PROVIDER=ollama, no key needed) ---
# OLLAMA_MODEL_NAME=gemma3:1b                 # chat model (default)
# OLLAMA_EMBEDDING_MODEL=mxbai-embed-large    # embedding model (default)
# OLLAMA_BASE_URL=http://localhost:11434      # Ollama server URL (default)

# --- Embeddings fallback (local, no key) ---
# Used when no provider key is available. Must match between ingest and query.
# HF_EMBEDDING_MODEL=all-MiniLM-L6-v2

# --- Vector Store Configuration ---
CHROMA_PERSIST_DIR=./chroma_db
```

🔐 A note on secrets: keys are read only from the environment / `.env` (which is
git-ignored). The app never prints key values. Cloud providers fail fast with a
clear message if their key is missing — switch to `LLM_PROVIDER=ollama` to run
fully local with no keys at all.

🌐 Switching to Local LLMs (Ollama)
To run fully local with zero API keys:
1. Install Ollama: Download and install the Ollama application for your OS.
2. Pull a chat model and an embedding model:
   ```
   ollama pull gemma3:1b
   ollama pull mxbai-embed-large
   ```
   (Override the defaults with `OLLAMA_MODEL_NAME` / `OLLAMA_EMBEDDING_MODEL` if you prefer other models, e.g. `ollama pull llama3`.)
3. Set the provider in your .env:
   ```LLM_PROVIDER=ollama```
4. Re-run `python ingest.py` so the vector store is built with the Ollama
   embeddings (embeddings must match between ingest and query).
__________________________________________________________________________________________________________________

🏃 How to Run
1. Ingest Data (First Time Only)
Ingest your test scripts or style examples (in plain text format) into the vector database. This powers the RAG retrieval.
```
python ingest.py
# Example output: Ingested 52 chunks into Chroma at ./chroma_db
```
2. Run the Interactive CLI
Start the interactive chat interface to generate code or ask general questions.
```python cli_chat.py```

