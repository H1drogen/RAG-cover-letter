from ingest import load_context_docs, split_docs

def main():

    docs = load_context_docs("data/add_your_data.txt")
    docs = split_docs(docs)
    print(docs)

if __name__ == "__main__":
    main()