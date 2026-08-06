from langchain import *
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_context_docs(doc_path: str) -> list[Document]:
    """Compile all documentation in the data directory as langchainDocuments."""
    paths = [doc_path]

    loader = TextLoader(doc_path)
    docs = loader.load()

    # for path in paths:
    #     url = f"{DOCS_BASE}/{path}.md"
    #     try:
    #         response = requests.get(url, timeout=20)
    #         response.raise_for_status()
    #     except requests.RequestException:
    #         continue
    #     source = f"{DOCS_BASE}/{path}"
    #     docs.append(
    #         Document(page_content=response.text, metadata={"source": source})
    #     )
    return docs


def split_docs(docs: list[Document]):
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    ).split_documents(docs)

    return chunks



