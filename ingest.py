#!/usr/bin/env python3
"""
Ingest plain text test script files into a vector DB with chunking.

Refactored to use LangChain's DirectoryLoader for standard document loading
and explicit Document objects for clear Chroma integration.
"""
import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Optional

load_dotenv()

# --- Imports for Document Loading ---
from langchain_community.document_loaders import DirectoryLoader, TextLoader
# --- Imports for Splitting and DB ---
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_chroma import Chroma
from langchain_core.documents import Document  # Explicit Document class

DEFAULT_DATA_DIR = os.getenv("DATA_DIR", "./data")
DEFAULT_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google").strip().lower()
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

EXTENSION_MAP = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".java": Language.JAVA,
    ".cpp": Language.CPP,
    ".go": Language.GO,
    ".rb": Language.RUBY,
    ".php": Language.PHP,
    ".rs": Language.RUST,
    ".scala": Language.SCALA,
    ".txt": Language.PYTHON, # Assumption: .txt usually contains Python in your workflow
}

def _huggingface_embeddings():
    """Local, zero-key fallback embeddings. Imported lazily."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings  # current package
    except ImportError:
        # Older installs still expose it under community.
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
    print(f"Using HuggingFaceEmbeddings (local): {HF_EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)


def create_embeddings():
    """Return an embedding model that matches the configured provider.

    IMPORTANT: the same embedding model must be used for both ingest and query,
    otherwise vector dimensions won't match. Keep LLM_PROVIDER consistent across
    `ingest.py` and the chat/driver runs.

    - ollama  -> local OllamaEmbeddings (zero keys)
    - openai  -> OpenAIEmbeddings (needs OPENAI_API_KEY)
    - google  -> GoogleGenerativeAIEmbeddings (needs GOOGLE_API_KEY)
    - anything else / missing deps -> local HuggingFace fallback (zero keys)
    """
    # --- LOCAL: Ollama (no key) ---
    if LLM_PROVIDER == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
            print(f"Using OllamaEmbeddings (local): {OLLAMA_EMBEDDING_MODEL}")
            return OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        except Exception as e:
            print(f"OllamaEmbeddings unavailable ({e}). Falling back to local HuggingFace.")
            return _huggingface_embeddings()

    # --- CLOUD: OpenAI ---
    if LLM_PROVIDER == "openai" and os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import OpenAIEmbeddings
            print("Using OpenAIEmbeddings.")
            return OpenAIEmbeddings()
        except Exception as e:
            print(f"OpenAIEmbeddings unavailable ({e}). Falling back to local HuggingFace.")
            return _huggingface_embeddings()

    # --- CLOUD: Google (Gemini API key, matches the chat provider) ---
    if LLM_PROVIDER == "google" and os.getenv("GOOGLE_API_KEY"):
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            print("Using GoogleGenerativeAIEmbeddings.")
            return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        except Exception as e:
            print(f"GoogleGenerativeAIEmbeddings unavailable ({e}). Falling back to local HuggingFace.")
            return _huggingface_embeddings()

    # --- LOCAL fallback (no key) ---
    return _huggingface_embeddings()

def get_splitter_for_file(filename: str, chunk_size: int, chunk_overlap: int):
    """Returns a text splitter optimized for the file's language."""
    _, ext = os.path.splitext(filename)
    lang = EXTENSION_MAP.get(ext.lower())
    
    if lang:
        return RecursiveCharacterTextSplitter.from_language(
            language=lang, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    else:
        # Fallback for unknown extensions (markdown, config files, etc.)
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

def ingest_folder(data_dir: str, persist_dir: str, chunk_size: int = 2000, chunk_overlap: int = 200):
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"{data_dir} not found.")

    print(f"Loading documents from {data_dir}...")

    # 2. Load ALL relevant extensions
    # Add any extension you plan to support here
    target_globs = ["**/*.py", "**/*.txt", "**/*.js", "**/*.ts", "**/*.java", "**/*.cpp", "**/*.go"]
    
    documents = []
    for glob_pattern in target_globs:
        loader = DirectoryLoader(
            data_dir,
            glob=glob_pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True}
        )
        try:
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            pass # Ignore empty globs

    if not documents:
        print("No files found.")
        return None

    # 3. Dynamic Chunking Loop
    print(f"Splitting {len(documents)} file(s) with language-specific logic...")
    
    final_chunks = []
    
    for doc in documents:
        source_file = doc.metadata.get("source", "")
        
        # Get the specific splitter for THIS file
        splitter = get_splitter_for_file(source_file, chunk_size, chunk_overlap)
        
        # Split just this one document
        chunks = splitter.split_documents([doc])
        final_chunks.extend(chunks)

    # 4. Ingest
    embeddings = create_embeddings()
    print(f"Ingesting {len(final_chunks)} chunks...")
    
    vectordb = Chroma.from_documents(
        documents=final_chunks, 
        embedding=embeddings,
        persist_directory=persist_dir
    )
    return vectordb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory with .txt files")
    parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR, help="Chroma persist directory")
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    args = parser.parse_args()
    ingest_folder(args.data_dir, args.persist_dir, args.chunk_size, args.chunk_overlap)
