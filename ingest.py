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

load_dotenv()

# --- Imports for Document Loading ---
from langchain_community.document_loaders import DirectoryLoader, TextLoader 
# --- Imports for Splitting and DB ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
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


DEFAULT_DATA_DIR = os.getenv("DATA_DIR", "./data")
DEFAULT_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").lower()


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

    # Fallback: local HuggingFace embeddings
    try:
        print("Using HuggingFaceEmbeddings.")
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        raise RuntimeError(f"Failed to instantiate HuggingFaceEmbeddings: {e}")


def ingest_folder(data_dir: str, persist_dir: str, chunk_size: int = 600, chunk_overlap: int = 100):
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"{data_dir} not found. Create it and add your .txt files.")

    # 1. Load documents using DirectoryLoader (Refactored L55/L62-L68)
    print(f"Loading documents from {data_dir}...")
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"} # <-- Fixed
    )
    docs: List[Document] = loader.load()

    if not docs:
        print("No .txt files found in", data_dir)
        return None

    # 2. Chunking
    print(f"Splitting {len(docs)} document(s) into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = splitter.split_documents(docs)

    # 3. Create Embeddings and Ingest (Refactored L80 to use split_docs directly)
    embeddings = create_embeddings()
    
    # Note: Chroma.from_documents expects a list of Document objects
    vectordb = Chroma.from_documents(
        documents=split_docs, 
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    # vectordb.persist() is deprecated in new Chroma, persistence happens on creation
    # but calling it doesn't hurt if you are on an older version.
    try:
        vectordb.persist()
    except Exception:
        pass # Ignore if not supported

    print(f"Ingested {len(split_docs)} chunks into Chroma at {persist_dir}")
    return vectordb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory with .txt files")
    parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR, help="Chroma persist directory")
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    args = parser.parse_args()
    ingest_folder(args.data_dir, args.persist_dir, args.chunk_size, args.chunk_overlap)