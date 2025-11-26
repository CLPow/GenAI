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
Install all required Python packages:
```
pip install -U python-dotenv pydantic langchain-core langchain-community langchain-chroma langchain-text-splitters langchain-google-genai sentence-transformers
```

3. Configure API Keys
Create a file named .env in the root directory and set your API keys and provider preference.
```
# --- LLM Provider Selection ---
# To use Google, set: LLM_PROVIDER=google
# To use Ollama (local), set: LLM_PROVIDER=ollama
# To use OpenAI, set: LLM_PROVIDER=openai
LLM_PROVIDER=google

# --- Google Gemini Configuration ---
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY_HERE
# GOOGLE_MODEL_NAME=gemini-2.5-flash (Optional override, defaults to this)

# --- OpenAI Configuration (Uncomment to use) ---
# OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE
# OPENAI_MODEL_NAME=gpt-4o-mini (Optional override)

# --- Vector Store Configuration ---
CHROMA_PERSIST_DIR=./chroma_db
```

🌐 Switching to Local LLMs (Ollama)
To run a Llama model locally, follow these steps:
1. Install Ollama: Download and install the Ollama application for your OS.
2. Download Model: Download a model like Llama 3 from your command line:
   ```ollama pull llama3```
3. Update .env: Change the provider setting in your .env file:
   ```LLM_PROVIDER=ollama```
The application will automatically use the llama3 model served by your local Ollama instance.
__________________________________________________________________________________________________________________

🏃 How to Run
1. Ingest Data (First Time Only)
Ingest your test scripts or style examples (in plain text format) into the vector database. This powers the RAG retrieval.
```
(venv) PS D:\AI> python ingest.py
# Example output: Ingested 52 chunks into Chroma at ./chroma_db
```
2. Run the Interactive CLI
Start the interactive chat interface to generate code or ask general questions.
```python cli_chat.py```
