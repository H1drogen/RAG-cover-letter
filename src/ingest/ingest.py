from langchain import *
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("data/add_your_data.txt")
docs = loader.load()

chunks = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
).split_documents(docs)

