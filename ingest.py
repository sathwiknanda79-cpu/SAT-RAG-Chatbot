from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import os


documents = []

folder = "documents"


for file in os.listdir(folder):

    if file.endswith(".pdf"):

        loader = PyPDFLoader(
            f"{folder}/{file}"
        )

        documents.extend(loader.load())


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)


chunks = splitter.split_documents(documents)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


db = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="vectorstore"
)


db.persist()


print("RAG database created successfully")