from langchain import *
from langchain_community.document_loaders import TextLoader

loader = TextLoader("data/add_your_data.txt")
docs = loader.load()
