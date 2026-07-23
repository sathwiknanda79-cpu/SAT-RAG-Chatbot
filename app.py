import hashlib
import html
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit_option_menu import option_menu


MODEL_NAME = "gemini-2.5-flash"


@st.cache_resource
def load_embedding() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_llm(api_key: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        api_key=api_key,
        temperature=0.2,
    )


def get_gemini_api_key() -> str | None:
    """Read the key from Streamlit secrets locally or from an environment variable."""
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return api_key

    try:
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        # Streamlit raises when no local secrets file has been created yet.
        return None


def build_vector_store(uploaded_file) -> Chroma:
    """Extract, split, and index one uploaded PDF in the current user session."""
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    temp_path = None

    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_path = temp_file.name

        documents = PyPDFLoader(temp_path).load()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        ).split_documents(documents)

        if not chunks:
            raise ValueError("No readable text was found in this PDF.")

        return Chroma.from_documents(chunks, load_embedding())
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="🤖",
    layout="wide",
)

st.markdown(
    """
    <style>
    body { background-color:#f7f5ff; }
    .main-title {
        font-size:45px;
        font-weight:800;
        background:linear-gradient(90deg,#6c2cff,#ff4fd8);
        -webkit-background-clip:text;
        color:transparent;
    }
    .card {
        background:white;
        padding:25px;
        border-radius:20px;
        box-shadow:0px 5px 20px #ddd;
        margin-bottom:20px;
    }
    .upload-box {
        border:3px dashed #7b3cff;
        padding:40px;
        border-radius:25px;
        text-align:center;
        background:#faf7ff;
    }
    .chat-user { background:#e7f0ff; padding:15px; border-radius:15px; }
    .chat-ai { background:#f0e7ff; padding:15px; border-radius:15px; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("🤖 RAG PDF Chatbot")
    selected = option_menu(
        menu_title=None,
        options=["Home", "Upload PDF", "Chat"],
        icons=["house", "file-earmark", "chat"],
        default_index=0,
    )
    st.info(
        """
        Built with:

        🦜 LangChain

        ✨ Gemini

        🗄️ ChromaDB

        ⚡ Streamlit
        """
    )

st.markdown(
    """
    <div class="card">
        <h1 class="main-title">Welcome to SAT RAG</h1>
        <h3 style="color:#7b3cff;">🤖 AI Document Assistant</h3>
        <p>Upload your PDF and ask questions. Gemini will answer from your document.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if selected == "Upload PDF":
    st.markdown(
        """
        <div class="upload-box">
            <h2>📄 Upload your PDF Document</h2>
            <p>AI will read and understand your file</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pdf = st.file_uploader("", type="pdf")
    if pdf:
        file_hash = hashlib.sha256(pdf.getvalue()).hexdigest()

        if st.session_state.get("uploaded_file_hash") != file_hash:
            try:
                with st.spinner("Reading and indexing your PDF..."):
                    st.session_state.vector_store = build_vector_store(pdf)
                    st.session_state.uploaded_file_hash = file_hash
                st.success("✅ PDF processed successfully")
            except Exception as error:
                st.session_state.pop("vector_store", None)
                st.error(f"Could not process this PDF: {error}")
        else:
            st.success("✅ This PDF is ready for questions")

st.markdown(
    """
    <div class="card">
        <h2>💬 Ask me a question</h2>
        <p>Ask anything from your uploaded PDF</p>
    </div>
    """,
    unsafe_allow_html=True,
)

question = st.chat_input("Ask me a question...")

if question:
    vector_store = st.session_state.get("vector_store")
    if vector_store is None:
        st.warning("Upload and process a PDF before asking a question.")
        st.stop()

    api_key = get_gemini_api_key()
    if not api_key:
        st.error(
            "Gemini is not configured. Add GEMINI_API_KEY to your Streamlit secrets "
            "or environment variables."
        )
        st.stop()

    try:
        docs = vector_store.similarity_search(question, k=4)
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = f"""You are a document assistant. Answer only from the supplied context.
If the answer is not in the context, say that you could not find it in the uploaded document.

Context:
{context}

Question:
{question}
"""

        with st.spinner("Gemini is preparing an answer..."):
            response = load_llm(api_key).invoke(prompt)

        answer = response.content
        if not isinstance(answer, str):
            answer = str(answer)
    except Exception as error:
        st.error(f"Could not generate an answer: {error}")
        st.stop()

    st.markdown(
        f"""
        <div class="chat-user">👤 You:<br>{html.escape(question)}</div>
        <br>
        <div class="chat-ai">🤖 SAT RAG:</div>
        """,
        unsafe_allow_html=True,
    )
    st.write(answer)
