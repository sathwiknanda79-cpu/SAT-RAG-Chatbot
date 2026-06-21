# SAT RAG 🤖

AI powered PDF chatbot using Retrieval Augmented Generation.

## Features

- Upload PDF documents
- Ask questions from documents
- Local LLM using Ollama
- Vector search using ChromaDB
- Beautiful Streamlit UI


## Tech Stack

Frontend:
- Streamlit

AI:
- Ollama
- Llama 3.2

RAG:
- LangChain
- ChromaDB

Embeddings:
- Sentence Transformers


## Architecture

PDF
↓
Text Extraction
↓
Chunking
↓
Embeddings
↓
Vector Database
↓
Retriever
↓
LLM
↓
Answer


## How to Run


Create environment:

python -m venv venv


Activate:

venv\Scripts\activate


Install:

pip install -r requirements.txt


Run:

streamlit run app.py
