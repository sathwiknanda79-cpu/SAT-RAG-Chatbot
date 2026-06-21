import streamlit as st
import os

from streamlit_option_menu import option_menu

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

@st.cache_resource
def load_embedding():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_llm():

    return Ollama(
        model="llama3.2:3b"
    )
# ---------- PAGE CONFIG ----------

st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="🤖",
    layout="wide"
)


# ---------- CUSTOM CSS ----------

st.markdown("""
<style>

body {
background-color:#f7f5ff;
}


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


.chat-user {

background:#e7f0ff;
padding:15px;
border-radius:15px;

}


.chat-ai {

background:#f0e7ff;
padding:15px;
border-radius:15px;

}

</style>

""",unsafe_allow_html=True)



# ---------- SIDEBAR ----------


with st.sidebar:

    st.title("🤖 RAG PDF Chatbot")

    selected = option_menu(
        menu_title=None,
        options=[
            "Home",
            "Upload PDF",
            "Chat"
        ],

        icons=[
            "house",
            "file-earmark",
            "chat"
        ],

        default_index=0
    )


    st.info(
        """
        Built with:

        🦜 LangChain

        🧠 Ollama

        🗄 ChromaDB

        ⚡ Streamlit
        """
    )



# ---------- HEADER ----------


st.markdown(
"""
<div class="card">

<h1 class="main-title">
Welcome to SAT RAG
</h1>

<h3 style="color:#7b3cff;">
🤖 AI Document Assistant
</h3>


<p>
Upload your PDF and ask questions.
Your local AI will answer from your document.
</p>


</div>

""",

unsafe_allow_html=True
)


# ---------- UPLOAD ----------


if selected=="Upload PDF":


    st.markdown(
    """
    <div class="upload-box">

    <h2>
    📄 Upload your PDF Document
    </h2>

    <p>
    AI will read and understand your file
    </p>

    </div>
    """,

    unsafe_allow_html=True
    )


    pdf = st.file_uploader(
        "",
        type="pdf"
    )


    if pdf:


        os.makedirs(
            "documents",
            exist_ok=True
        )


        path=f"documents/{pdf.name}"


        with open(path,"wb") as f:

            f.write(pdf.getbuffer())


        loader=PyPDFLoader(path)

        docs=loader.load()


        splitter=RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )


        chunks=splitter.split_documents(docs)


        embeddings=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
        )


        Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory="vectorstore"
        )


        st.success(
            "✅ PDF processed successfully"
        )




# ---------- CHAT BOX BELOW UPLOAD ----------


st.markdown(
"""
<div class="card">

<h2>
💬 Ask me a question
</h2>

<p>
Ask anything from your uploaded PDF
</p>

</div>

""",

unsafe_allow_html=True
)


question = st.chat_input(
    "Ask me a question..."
)



if question:

    embeddings = load_embedding()


    db = Chroma(
        persist_directory="vectorstore",
        embedding_function=embeddings
    )


    docs = db.similarity_search(question)


    context = "\n".join(
        [doc.page_content for doc in docs]
    )


    llm = load_llm()


    prompt = f"""

Answer only using this:

{context}


Question:

{question}

"""


    answer = llm.invoke(prompt)


    st.markdown(
    f"""
    <div class="chat-user">

    👤 You:

    {question}

    </div>

    <br>

    <div class="chat-ai">

    🤖 SAT RAG:

    </div>
    """,
    unsafe_allow_html=True
    )


    st.write(answer)