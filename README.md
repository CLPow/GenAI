# 📄 Project Readme: Multipurpose RAG GenAI System

🎯 Overview
This project is a general-purpose Retrieval-Augmented Generation (RAG) system. It answers and generates **grounded in whatever dataset you ingest** — code, docs, CSVs, notes, anything — and its behavior is shaped entirely by a **user-defined system prompt**. Set that prompt and the same app becomes an internal wiki, an information desk, a data analyst, a code writer, or a Q&A bot. It uses a Chroma vector database for retrieval and supports Google Gemini, OpenAI, and local models via Ollama.

Key Features:

You Define the Purpose: The assistant's role is never hard-coded. Set it via `SYSTEM_PROMPT` / `SYSTEM_PROMPT_FILE`, or change it live with the `/system` command.

Grounded Answers: Every query retrieves relevant chunks from your knowledge base and answers from that CONTEXT.

On-Demand Code Generation: The `/code` command produces structured, runnable files (parsed into Pydantic models) when you want artifacts instead of prose.

Flexible Deployment: Easily switch between cloud APIs and local LLMs.
__________________________________________________________________________________________________________________

⚙️ Setup and Installation
This project requires Python 3.11 or higher due to modern dependency requirements (LangChain V0.2+ and Pydantic V2).

1. Clone the Repository and Set Up the Environment
```
# Clone the repository
git clone https://github.com/CLPow/GenAI.git
cd GenAI

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
The quickest path is the pinned manifest, which installs the core packages plus
`pandas` (used by `csv_ingest.py`):
```
pip install -U -r requirements.txt
```

Or install the core packages by hand (always required):
```
pip install -U python-dotenv pydantic langchain-core langchain-community langchain-chroma langchain-text-splitters langchain-huggingface sentence-transformers
```

Either way, install ONLY the provider packages you intend to use. SDKs are
imported lazily per-provider, so a missing one never breaks the others (the
provider lines in `requirements.txt` are commented out for the same reason):
```
# Google Gemini (cloud)
pip install -U langchain-google-genai

# OpenAI (cloud)
pip install -U langchain-openai

# Ollama (local, no API key required)
pip install -U langchain-ollama
```

3. Configure Provider and Keys
Copy the template to a real `.env` (which is git-ignored), then edit it:
```
# Windows (PowerShell): copy .env.example .env
# Linux/macOS:          cp .env.example .env
```
`LLM_PROVIDER` is the single
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

# --- Assistant role (the "purpose") -- you define it ---
# Precedence: SYSTEM_PROMPT_FILE > SYSTEM_PROMPT > built-in generic default.
# SYSTEM_PROMPT=You are an internal company wiki. Answer strictly from CONTEXT and cite the source File.
# SYSTEM_PROMPT_FILE=./system_prompt.txt
# RAG_K=4   # number of context chunks retrieved per query (default 4)

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
1. Ingest Your Data (First Time Only)
Drop your knowledge base into `./data` (override with `DATA_DIR`) and ingest it.
`ingest.py` walks the folder, sends each file to a format-specific loader, and
chunks the extracted text with language-aware splitting.

Supported inputs:
| Category            | Extensions                                                        | Optional dependency |
|---------------------|-------------------------------------------------------------------|---------------------|
| Text / code / data  | `.py .js .ts .java .cpp .go .rb .php .rs .scala .txt .md .rst .json .csv .yaml/.yml .html .css .sql` | (built-in)          |
| PDF                 | `.pdf`                                                             | `pypdf`             |
| Word                | `.docx`                                                           | `docx2txt`          |
| Excel               | `.xlsx`, `.xls`                                                   | `openpyxl` / `xlrd` |
| PowerPoint          | `.pptx`                                                           | `python-pptx`       |
| Images (OCR)        | `.png .jpg .jpeg .tif .tiff .bmp .gif .webp`                      | `pytesseract` + `Pillow` + Tesseract engine |

Loaders import their dependency **lazily**, so a missing one disables only that
file type (with a clear pip hint) — the rest still ingest. Install just the
formats you need (see the optional lines in `requirements.txt`).

**Image OCR** additionally needs the Tesseract engine installed on your system
(`winget install UB-Mannheim.TesseractOCR` on Windows, `apt install tesseract-ocr`
on Debian/Ubuntu, `brew install tesseract` on macOS). If it isn't on your PATH,
set `TESSERACT_CMD` in `.env`. Legacy `.doc`/`.ppt` are not supported — convert
them to `.docx`/`.pptx` first.

```
python ingest.py
# + report.pdf (12 section(s))
# + sales.xlsx (3 section(s))
# + deck.pptx (8 section(s))
# Ingesting 52 chunks...
```

2. Define the Purpose (the system prompt)
The assistant has no preset role. Choose one of:
- `SYSTEM_PROMPT="..."` in `.env` (inline), or
- `SYSTEM_PROMPT_FILE=./system_prompt.txt` (a file), or
- set it live in the CLI: `/system You are an internal wiki; answer only from CONTEXT.`

If none is set, a generic "answer from your knowledge base" prompt is used.

3. Run the Interactive CLI
```python cli_chat.py```

By default every request is answered with RAG (retrieve → answer in your role).
Commands:
| Command           | What it does                                            |
|-------------------|---------------------------------------------------------|
| `<anything>`      | Ask a question, answered from your ingested data         |
| `/code <request>` | Generate structured, runnable file(s) for the request   |
| `/system`         | Show the current role                                    |
| `/system <text>`  | Set a new role for this session                          |
| `/reset`          | Clear the conversation memory                            |
| `/help`           | List commands                                            |
| `quit` / `exit`   | Leave (memory is already saved each turn)               |

Conversation history persists to `chat_history.json`.
__________________________________________________________________________________________________________________

🧠 Memory (durable · model-agnostic · self-compacting)
Conversation memory (`memory.py`) is built to stay reliable as sessions get long
and to behave consistently even if you switch models:
- **Durable:** saved after *every* turn with an atomic write (temp file +
  replace), so a crash or hard-close never loses or corrupts memory.
- **Model-agnostic / shared:** stored as a natural-language summary plus plain
  message records — nothing provider-specific. The same `chat_history.json`
  works across Gemini / OpenAI / Ollama, giving consistent recollection when you
  swap models. (Note: this applies to chat memory; the *vector store* must still
  be ingested with one consistent embedding model.)
- **Self-compacting:** when recent messages exceed `MEMORY_MAX_MESSAGES`, the
  oldest are folded into a rolling summary (keeping the last `MEMORY_KEEP_RECENT`
  verbatim), so prompts stay small and the assistant still remembers what it did.

🔐 Safety
- **Prompt-injection guard:** retrieved CONTEXT is wrapped in clear delimiters
  and a fixed system rail tells the model to treat it as untrusted data, never
  as instructions — so text planted in your dataset can't hijack the assistant.
- **Sandboxed file writes:** `/code` saves are confined to `./generated_tests/`
  via filename sanitization, an extension allowlist (no shell/batch types), a
  path-escape check, and a per-file size cap (`MAX_GENERATED_FILE_BYTES`).
- **Input cap:** requests are limited to `MAX_INPUT_CHARS`.
- **Secrets:** keys are read only from the environment / git-ignored `.env` and
  are never printed.
__________________________________________________________________________________________________________________

🧩 Optional: Inspecting a CSV
`csv_ingest.py` is a small standalone helper (requires `pandas`) for loading and
sanity-checking a CSV — schema, null counts — before you feed data into a
pipeline. It is not wired into the RAG flow; use it directly:
```python
from csv_ingest import CSVIngest

ci = CSVIngest("path/to/file.csv")
ci.load_data()
ci.analyze_schema()
ci.detect_missing_attachments()
```

