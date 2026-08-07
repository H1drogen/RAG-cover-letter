from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_aws import BedrockEmbeddings

EmbeddingProvider = Literal["openai", "bedrock"]

@dataclass(frozen=True)
class EmbeddingConfig:
    provider: EmbeddingProvider = "openai"

    openai_model: str = "text-embedding-3-small"
    openai_api_key: str | None = None
    openai_base_url: str | None = None

    bedrock_model_id: str = "amazon.titan-embed-text-v1"
    bedrock_region_name: str | None = None
    bedrock_profile_name: str | None = None


@lru_cache(maxsize=None)
def get_embeddings(config: EmbeddingConfig = EmbeddingConfig()) -> Embeddings:
    if config.provider == "openai":
        return OpenAIEmbeddings(
            model=config.openai_model,
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
        )

    if config.provider == "bedrock":
        return BedrockEmbeddings(
            model_id=config.bedrock_model_id,
            region_name=config.bedrock_region_name,
            credentials_profile_name=config.bedrock_profile_name,
        )

    raise ValueError(f"Unsupported embedding provider: {config.provider}")