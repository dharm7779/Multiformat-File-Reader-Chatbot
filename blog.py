import streamlit as st
from pdf_qa import PdfQA
from pathlib import Path
from tempfile import NamedTemporaryFile
import time
import shutil
from constants import *


# Create a dictionary to map file extensions to document types
DOCUMENT_TYPES = {
    ".pdf": "pdf",
    ".pptx": "pptx",
    ".xlsx": "xlsx",
    ".docx": "docx",
}

# Streamlit app code
st.set_page_config(
    page_title='MultiFormat File Reader Chatbot',
    page_icon='🔖',
    layout='wide',
    initial_sidebar_state='auto',
)
st.markdown("<h1 style='text-align: center; color: blue;'>MultiFormat File Reader Chatbot📄 </h1>",
            unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: B;'>Built by Dharm Sagparia & Abhimanyu Gupta </a></h3>",
            unsafe_allow_html=True)

if "pdf_qa_model" not in st.session_state:
    st.session_state["pdf_qa_model"]: PdfQA = PdfQA()  # Initialization

# To cache resource across multiple sessions


@st.cache_resource
def load_llm(llm, load_in_8bit):

    if llm == LLM_OPENAI_GPT35:
        pass
    elif llm == LLM_FLAN_T5_SMALL:
        return PdfQA.create_flan_t5_small(load_in_8bit)
    elif llm == LLM_FLAN_T5_BASE:
        return PdfQA.create_flan_t5_base(load_in_8bit)
    elif llm == LLM_FLAN_T5_LARGE:
        return PdfQA.create_flan_t5_large(load_in_8bit)
    elif llm == LLM_FASTCHAT_T5_XL:
        return PdfQA.create_fastchat_t5_xl(load_in_8bit)
    elif llm == LLM_FALCON_SMALL:
        return PdfQA.create_falcon_instruct_small(load_in_8bit)
    else:
        raise ValueError("Invalid LLM setting")

# To cache resource across multiple sessions


@st.cache_resource
def load_emb(emb):
    if emb == EMB_INSTRUCTOR_XL:
        return PdfQA.create_instructor_xl()
    elif emb == EMB_SBERT_MPNET_BASE:
        return PdfQA.create_sbert_mpnet()
    elif emb == EMB_SBERT_MINILM:
        pass  # ChromaDB takes care
    else:
        raise ValueError("Invalid embedding setting")


with st.sidebar:
    emb = st.radio("**Select Embedding Model**",
                   [EMB_INSTRUCTOR_XL, EMB_SBERT_MPNET_BASE, ], index=1)  # EMB_SBERT_MINILM
    llm = st.radio("**Select LLM Model**", [LLM_FASTCHAT_T5_XL, LLM_FLAN_T5_SMALL,
                   LLM_FLAN_T5_BASE, LLM_FLAN_T5_LARGE, LLM_FLAN_T5_XL], index=2)
    load_in_8bit = st.radio("**Load 8 bit**", [True, False], index=1)
    uploaded_file = st.file_uploader(
        "**Upload Document**", type=["pdf", "pptx", "xlsx", "docx"])

    if st.button("Submit") and uploaded_file:
        with st.spinner(text="Uploading Document and Generating Embeddings.."):
            # Determine the document type based on file extension
            file_extension = Path(uploaded_file.name).suffix
            document_type = DOCUMENT_TYPES.get(file_extension)

            if document_type:
                temp_path = NamedTemporaryFile(
                    delete=False, suffix=file_extension)
                shutil.copyfileobj(uploaded_file, temp_path)
                temp_path = Path(temp_path.name)

                st.session_state["pdf_qa_model"].config = {
                    "document_type": document_type,
                    "document_path": str(temp_path),
                    "embedding": emb,
                    "llm": llm,
                    "load_in_8bit": load_in_8bit
                }
                st.session_state["pdf_qa_model"].embedding = load_emb(emb)
                st.session_state["pdf_qa_model"].llm = load_llm(
                    llm, load_in_8bit)
                st.session_state["pdf_qa_model"].init_embeddings()
                st.session_state["pdf_qa_model"].init_models()
                st.session_state["pdf_qa_model"].vector_db_documents(
                    document_type)
                st.sidebar.success("Document uploaded successfully")

question = st.text_input('Ask a question', 'What is this document?')

if st.button("Answer"):
    try:
        st.session_state["pdf_qa_model"].retreival_qa_chain()
        answer = st.session_state["pdf_qa_model"].answer_query(question)
        st.write(f"{answer}")
    except Exception as e:
        st.error(f"Error answering the question: {str(e)}")
