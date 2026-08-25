# src/indexing/embed/vectorstore.py

from functools import lru_cache
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools import tool


def create_retriever(doc_splits, embedding, k: int = 5):
    """Create vector store with semantic search."""
    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits,
        embedding=embedding,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})


@tool
def retrieve_content(query: str, retriever) -> str:
    """Search and return relevant chunks with metadata."""
    retrieved_docs = retriever.invoke(query)

    formatted = []
    for doc in retrieved_docs:
        skills = doc.metadata.get('technical_skills', [])
        soft = doc.metadata.get('soft_skills', [])
        impact = doc.metadata.get('has_quantified_impact', False)

        header = f"[{doc.metadata.get('achievement_type', 'EXPERIENCE').upper()}]"
        if skills:
            header += f" | Skills: {', '.join(skills)}"
        if impact:
            header += " | ✓ Quantified Impact"

        formatted.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted)