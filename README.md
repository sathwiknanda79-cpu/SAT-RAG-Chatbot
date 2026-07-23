# SAT RAG

An AI-powered PDF chatbot using retrieval-augmented generation (RAG).

## Features

- Upload PDF documents
- Ask questions grounded in the uploaded document
- Gemini API with your own key
- Vector search with ChromaDB
- Streamlit user interface

## Technology

- Streamlit
- Google Gemini
- LangChain
- ChromaDB
- Sentence Transformers embeddings

## Run locally

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set your Gemini API key for the current PowerShell session:

```powershell
$env:GEMINI_API_KEY="your-api-key"
```

Start the app:

```powershell
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this project to GitHub. Do not upload `venv/` or your API key.
2. Create an app at https://share.streamlit.io and select `app.py`.
3. In **Advanced settings → Secrets**, add:

```toml
GEMINI_API_KEY = "your-api-key"
```

The deployed app receives a public `streamlit.app` URL and does not require your local computer or Ollama server to be running.
