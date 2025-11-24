#!/usr/bin/env python3
"""
Ingest plain text test script files into a vector DB with chunking.

This version prefers local HuggingFace embeddings as a reliable fallback.
If you have Vertex / Google embeddings set up (service account + LangChain VertexAIEmbeddings),
the script will try to use them when LLM_PROVIDER=google and GOOGLE_APPLICATION_CREDENTIALS is set.
"""
import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DEFAULT_DATA_DIR = os.getenv("DATA_DIR", "./data")
DEFAULT_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").lower()

# LangChain imports (ensure requirements installed)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# Embeddings imports (we'll try Vertex/Google if available, otherwise HuggingFace local)
try:
    # VertexAIEmbeddings is available when using Google Vertex via LangChain
    from langchain.embeddings import VertexAIEmbeddings  # type: ignore
except Exception:
    VertexAIEmbeddings = None

from langchain.embeddings import HuggingFaceEmbeddings
try:
    from langchain.embeddings import OpenAIEmbeddings  # optional
except Exception:
    OpenAIEmbeddings = None


def create_embeddings():
    # If provider is google and VertexAIEmbeddings available and GOOGLE_APPLICATION_CREDENTIALS is set, use Vertex
    if LLM_PROVIDER == "google" and VertexAIEmbeddings and os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        try:
            return VertexAIEmbeddings()
        except Exception as e:
            print("VertexAIEmbeddings import succeeded but instantiation failed:", e)
            print("Falling back to HuggingFaceEmbeddings.")
    # If OPENAI_API_KEY present and OpenAIEmbeddings available, you could use it (optional)
    if os.getenv("OPENAI_API_KEY") and OpenAIEmbeddings is not None:
        try:
            return OpenAIEmbeddings()
        except Exception:
            pass
    # Fallback: local HuggingFace embeddings (no external quota)
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def ingest_folder(data_dir: str, persist_dir: str, chunk_size: int = 600, chunk_overlap: int = 100):
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"{data_dir} not found. Create it and add your .txt files.")

    docs = []
    for p in data_path.rglob("*.txt"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        metadata = {"source": str(p), "filename": p.name}
        docs.append({"page_content": text, "metadata": metadata})

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_texts = []
    for d in docs:
        chunks = splitter.split_text(d["page_content"])
        for idx, c in enumerate(chunks):
            md = d["metadata"].copy()
            md["chunk"] = idx
            split_texts.append({"page_content": c, "metadata": md})

    if not split_texts:
        print("No .txt files found in", data_dir)
        return None

    embeddings = create_embeddings()
    texts = [s["page_content"] for s in split_texts]
    metadatas = [s["metadata"] for s in split_texts]

    vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, metadatas=metadatas, persist_directory=persist_dir)
    vectordb.persist()
    print(f"Ingested {len(texts)} chunks into Chroma at {persist_dir}")
    return vectordb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory with .txt files")
    parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR, help="Chroma persist directory")
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    args = parser.parse_args()
    ingest_folder(args.data_dir, args.persist_dir, args.chunk_size, args.chunk_overlap)