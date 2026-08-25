import re
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

TECH_SKILLS = {
    'python', 'aws', 's3', 'boto3', 'react', 'javascript', 'typescript',
    'kubernetes', 'docker', 'api', 'sql', 'redis', 'microservices',
    'cloudwatch', 'debugging', 'git', 'ci/cd', 'sdk', 'sdk'
}

SOFT_SKILLS = {
    'leadership', 'teamwork', 'communication', 'problem-solving',
    'resilience', 'collaboration', 'analytical', 'strategic thinking'
}

def load_context_docs(doc_path: str) -> list[Document]:
    """Load every text file recursively from a path."""
    root = Path(doc_path)
    if root.is_file():
        return TextLoader(str(root)).load()

    docs: list[Document] = []
    for file_path in sorted(p for p in root.rglob("*") if p.is_file()):
        try:
            docs.extend(TextLoader(file_path).load())
        except Exception:
            continue
    return docs


def extract_metadata(text: str) -> dict:
    """Extract skills, keywords, and competency type from text."""
    text_lower = text.lower()

    skills = [s for s in TECH_SKILLS if s in text_lower]
    soft_skills = [s for s in SOFT_SKILLS if s in text_lower]

    has_impact = any(
        word in text_lower
        for word in ['won', 'achieved', 'delivered', 'improved', 'reduced',
                     'increased', 'first place', 'finalist', 'silver', 'gold']
    )

    achievement_type = 'technical' if skills else ('leadership' if any(w in text_lower for w in ['team', 'president', 'led', 'managed']) else 'general')

    return {
        'technical_skills': skills,
        'soft_skills': soft_skills,
        'has_quantified_impact': has_impact,
        'achievement_type': achievement_type,
    }


def split_docs(docs: list[Document]) -> list[Document]:
    """
    Semantically split documents at section boundaries (achievements/experiences).
    Chunks at 1800 chars with 200 char overlap. Extracts metadata for better RAG retrieval.
    """
    chunked_docs = []

    for doc in docs:
        text = doc.page_content

        # Split at section headers (lines starting with capital letter, no bullet)
        # This keeps related achievements together
        sections = re.split(r'\n(?=[A-Z][a-zA-Z\s]+)\n', text)

        for section in sections:
            if len(section.strip()) < 100:
                continue

            # Keep small sections whole
            if len(section) <= 1800:
                metadata = extract_metadata(section)
                metadata.update(doc.metadata or {})

                chunked_docs.append(Document(
                    page_content=section.strip(),
                    metadata=metadata
                ))
            else:
                # Split larger sections while preserving context
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1800,
                    chunk_overlap=200,
                )
                sub_chunks = splitter.split_text(section)

                for sub_chunk in sub_chunks:
                    if len(sub_chunk.strip()) > 50:
                        metadata = extract_metadata(sub_chunk)
                        metadata.update(doc.metadata or {})

                        chunked_docs.append(Document(
                            page_content=sub_chunk.strip(),
                            metadata=metadata
                        ))

    return chunked_docs


