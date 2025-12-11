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
from langchain_core.documents import Document # Explicit Document class

# --- Imports for Embeddings ---
try:
    from langchain_community.embeddings import VertexAIEmbeddings  # type: ignore
except Exception:
    VertexAIEmbeddings = None

from langchain_community.embeddings import HuggingFaceEmbeddings
try:
    from langchain_community.embeddings import OpenAIEmbeddings  # optional
except Exception:
    OpenAIEmbeddings = None
try:
    from langchain_community.embeddings import OllamaEmbeddings
except Exception:
    OllamaEmbeddings = None


DEFAULT_DATA_DIR = os.getenv("DATA_DIR", "./data")
DEFAULT_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").lower()
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

def create_embeddings():
    """Factory that returns an embedding model instance."""
    if LLM_PROVIDER == "google" and VertexAIEmbeddings and os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        try:
            print("Using VertexAIEmbeddings.")
            return VertexAIEmbeddings()
        except Exception as e:
            print("VertexAIEmbeddings instantiation failed:", e)
            print("Falling back to HuggingFaceEmbeddings.")

    if os.getenv("OPENAI_API_KEY") and OpenAIEmbeddings is not None:
        try:
            print("Using OpenAIEmbeddings.")
            return OpenAIEmbeddings()
        except Exception:
            pass

    # NEW: Ollama Check (for local RAG consistency)
    if LLM_PROVIDER == "ollama" and OllamaEmbeddings is not None:
        try:
            print(f"Using OllamaEmbeddings with model: {OLLAMA_EMBEDDING_MODEL}")
            # Ensure the model is pulled in Ollama (e.g., ollama pull mxbai-embed-large)
            return OllamaEmbeddings(
                model=OLLAMA_EMBEDDING_MODEL,
                base_url=OLLAMA_BASE_URL
            )
        except Exception as e:
            print(f"OllamaEmbeddings instantiation failed: {e}")
            print("Falling back to HuggingFaceEmbeddings.")

    # Fallback: local HuggingFace embeddings
    try:
        print("Using HuggingFaceEmbeddings.")
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        raise RuntimeError(f"Failed to instantiate HuggingFaceEmbeddings: {e}")

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
