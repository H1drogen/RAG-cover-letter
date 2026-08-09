from indexing.ingest import load_context_docs, split_docs
from indexing.embed import get_embeddings, EmbeddingConfig

def main():

    docs = load_context_docs("data/")
    docs = split_docs(docs)


    config = EmbeddingConfig(
        'bedrock',
        "text-embedding-3-small", None, None,
        "amazon.titan-embed-text-v1", None, None
    )
    get_embeddings(config)

    print(docs)



if __name__ == "__main__":
    main()