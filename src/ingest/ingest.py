from pathlib import Path

from langchain import *
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_context_docs(doc_path: str) -> list[Document]:
    """Load every text file under a specified path recursively."""
    root = Path(doc_path)
    if root.is_file():
        return TextLoader(str(root)).load()

    docs: list[Document] = []
    for file_path in sorted(p for p in root.rglob("*") if p.is_file()):
        docs.extend(TextLoader(file_path).load())

    return docs

def split_docs(docs: list[Document]):
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    ).split_documents(docs)

    return chunks



