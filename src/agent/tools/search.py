import uuid

from deepagents.backends import StateBackend
from langchain.tools import tool

from indexing.embed import vectorstore

backend = StateBackend()


@tool(parse_docstring=True)
def search_content(query: str) -> str:
    """Search documentation and save matching chunks to the agent filesystem.

    Args:
        query: Natural language search query.

    Returns:
        File paths where retrieved chunks were saved under /retrieved/.
    """
    retrieved_docs = vectorstore.retrieve_content(query, k=4)
    batch_id = uuid.uuid4().hex[:8]
    uploads: list[tuple[str, bytes]] = []
    saved_paths: list[str] = []

    for index, doc in enumerate(retrieved_docs, start=1):
        path = f"/retrieved/{batch_id}/chunk_{index}.md"
        content = (
            f"# Source: {doc.metadata.get('source', 'unknown')}\n\n"
            f"{doc.page_content}"
        )
        uploads.append((path, content.encode("utf-8")))
        saved_paths.append(path)

    backend.upload_files(uploads)
    return (
        f"Saved {len(saved_paths)} documentation chunks:\n"
        + "\n".join(saved_paths)
    )