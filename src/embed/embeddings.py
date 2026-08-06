from functools import lru_cache
from dataclasses import dataclass

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings

@dataclass(frozen=True)
class OpenAIEmbeddingConfig:
    model: str = "text-embedding-3-small"
    api_key: str | None = None
    base_url: str | None = None


@lru_cache(maxsize=1)
def get_embeddings(config: OpenAIEmbeddingConfig = OpenAIEmbeddingConfig()) -> Embeddings:
    return OpenAIEmbeddings(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
    )