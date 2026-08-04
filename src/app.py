from ingest import load_context_docs, split_docs

docs = load_context_docs("data/add_your_data.txt")
docs = split_docs(docs)
print(docs)