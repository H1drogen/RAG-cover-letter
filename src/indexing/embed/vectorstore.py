from functools import lru_cache

from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools import tool


@lru_cache(maxsize=1)
def create_retriever(doc_splits, embedding):
    """Create the vector store."""
    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits,
        embedding=embedding,
    )
    return vectorstore.as_retriever()


@tool
def retrieve_content(query: str, retriever) -> str:
    """Search and return information from the vector store."""
    retrieved_docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in retrieved_docs])