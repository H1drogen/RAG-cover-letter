import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_aws import BedrockEmbeddings


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: Literal["openai", "bedrock"]

    openai_api_key: str | None
    openai_base_url:  str | None
    openai_model: str


    bedrock_region_name: str | None
    bedrock_profile_name: str | None
    bedrock_model_id: str


@lru_cache(maxsize=None)
def get_embeddings(config: EmbeddingConfig) -> Embeddings:
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