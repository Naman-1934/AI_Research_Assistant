import streamlit as st 
import numpy as np
import shutil
import os 

from utils.pdf_loader import extract_text_from_Pdfs
from utils.text_splitter import split_text
from utils.embeddings import load_embedding_model
from utils.vector_stores import create_faiss_index, save_vector_store, load_vector_store
from utils.rag_chain import get_llm, generate_answer, generate_summary
from utils.retriever import retrieve_relevant_chunks
from utils.validators import validators_api_key, validate_uploaded_files, validate_question, validate_vector_store
from utils.document_validator import is_research_paper


# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(page_title="AI Research Assistant", page_icon="📚", layout="wide", initial_sidebar_state="expanded")
st.title("📚 AI Research Assistant")
st.caption("Ask questions, generate summaries, and explores research papers using Gemini + FAISS")

# ──────────────────────────────────────────────
# Load all the  model which will be used in this project.
# ──────────────────────────────────────────────
llm = get_llm()
embedding_model = load_embedding_model()


# ──────────────────────────────────────────────
# LOAD SAVED VECTOR DB FROM DISK
# If user already processed PDFs in a previous session,
# load that saved index and chunks automatically.
# Returns (None, None) if no saved DB exists yet.
# ──────────────────────────────────────────────
index, chunks = load_vector_store()
if  index is not None:
    st.session_state.vector_store = index

# ──────────────────────────────────────────────
# SESSION STATE — chat history
# st.session_state persists across reruns in the same session.
# Without this, chat history would reset every time the user types.
# ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# ──────────────────────────────────────────────
# SIDEBAR
# New Chat
# ──────────────────────────────────────────────
with st.sidebar:
    if st.button("New Chat"):

        # Clear chat 
        st.session_state.messages = []

        # Remove current vector store
        st.session_state.vector_store = None

        # Change uploader key 
        st.session_state.uploader_key += 1

        # Clear local variable
        index = None
        chunks = []

        # Delete saved FAISS database
        if os.path.exists("vector_db"):
            shutil.rmtree("vector_db")

        st.success("Started a new conversation.")
        st.rerun()

# ──────────────────────────────────────────────
# SIDEBAR
# Clear Chat
# ──────────────────────────────────────────────
with st.sidebar:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.success("Chat history cleared.")
        st.rerun()

# ──────────────────────────────────────────────
# Reset Entire Project
# ──────────────────────────────────────────────
if st.button("Reset Project"):
    st.session_state.messages = []
    st.session_state.vector_store = None
    index = None

    chunks = []

    # Delete saved FAISS files
    if os.path.exists("vector_db"):
        shutil.rmtree("vector_db")
    st.success("Project has been reset.")
    st.rerun()

# ──────────────────────────────────────────────
# STATUS BANNER
# Tells the user clearly what state the app is in.
# ──────────────────────────────────────────────
    st.markdown("""
Welcome to AI Research Assistant
                
This web application allows you to:
- 📄 Upload one or more research papers
- 🤖 Ask questions using AI
- 📝 Generate document summaries
- 🔍 Retrieve information from PDFs
- 📚 View source references
                
🚀 Getting Started
1. Upload a research paper.
2. Wait for processing.
3. Ask questions in the chat.
"""
    )


# ──────────────────────────────────────────────
# CHAT HISTORY DISPLAY
# FIX Bug 3 (from Part A): only ONE render loop here.
# The second duplicate loop that was below the file uploader is removed.
# ──────────────────────────────────────────────
for messages in st.session_state.messages:
    with st.chat_message(messages['role']):
        st.markdown(messages['content'])


# ──────────────────────────────────────────────
# PDF UPLOAD
# ──────────────────────────────────────────────
uploaded_file = st.file_uploader("📄 Upload Research Paper (PDF)", type="pdf", accept_multiple_files=False)

if uploaded_file:
    st.success(f"📄 Selected File: {uploaded_file.name} ")


# ──────────────────────────────────────────────
# PDF PROCESSING
# Only runs when user actually uploads files.
# All 4 pipeline steps happen here:
# extract → chunk → embed → save to disk
# ──────────────────────────────────────────────
if validate_uploaded_files(uploaded_file):
    with st.spinner("📄 Processing research paper..."):

        # Step 1 — extract raw text from all uploaded PDFs
        raw_text = extract_text_from_Pdfs([uploaded_file])

        # ------------------------------------
        # Validate uploaded document
        # ------------------------------------
        is_valid, validatation_message = is_research_paper(raw_text)

        if not is_valid:
            st.error("❌ Unsupported Document")
            st.warning(validatation_message)
            st.info(
                """
            This application only supports academic research papers.

            Supported:
            IEEE Papers,
            Springer Papers,
            Elsevier Papers,
            ACM Papers,
            University Research Papers,

            Unsupported:
            Question Papers,
            Exam Papers,
            News Articles,
            Resume/CV,
            Bills,
            Certificates
            """
            )
            st.stop()

        # Step 2 — split into 1000-char chunks with 200-char overlap
        chunks = split_text(raw_text)

        # Step 3 — embed all chunks and build FAISS index
        index, vector_store = create_faiss_index(chunks, embedding_model)
        st.session_state.vector_store = vector_store

        # Step 4 — save index + chunks to disk so they survive app restarts
        save_vector_store(index, chunks)

        st.session_state.vector_store = index

        st.success("✅ Research paper processed successfully.")


    # ── SUMMARY SECTION ──────────────────────────────────────
    # Kept strictly inside `if uploaded_files:` because raw_text
    # only exists here. Moving it outside would cause NameError.
    # ─────────────────────────────────────────────────────────
    if st.button("📝 Generate Document Summary"):
        with st.spinner("📝 Generate Research Paper Summary"):
            # First 30,000 chars only — keeps us inside Gemini token limits
            summary = generate_summary(llm, raw_text[:85000])

        st.subheader("📑 Research Paper Summary")
        st.write(summary)

        # FIX Bug 4: was mine="text/plain" — typo, silent failure
        # Corrected to mime="text/plain"
        st.download_button(label=" Download Summary", data=summary, file_name="summary.txt", mime="text/plain")


# ──────────────────────────────────────────────
# Process User Question
# This creates the chatbot input field at the bottom.
# ──────────────────────────────────────────────
user_question = st.chat_input("💬 Ask anything about from the  uploaded research paper...")

if (validate_question(user_question) and validate_vector_store(st.session_state.vector_store)):

    # Save User Message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        
        with st.spinner("Searching documents..."):
            
            try:

            # PDF contains:
            # Chapter 1: Objective of Research, Chapter 2: Methodology, Chapter 3:Results,
            # Question: "What is the objective?" and Retriever returns:
            # [ "The objective of this research is..."]
                relevant_results = retrieve_relevant_chunks(index, chunks, user_question, top_k=5)

                if not relevant_results:
                    answer = (
                        "I Couldn't find relevant information in the document."
                    )

                else:

                    context = "\n\n".join(
                        [
                            item["content"]
                            for item in relevant_results
                        ]
                    )

                    # Question: What is the objective?
                    # Context: Objective of the research is...
                    answer = generate_answer(llm = llm, question=user_question, context=context, chat_history=st.session_state.messages)

                    #  Where did you find this answer so, we will return chunk_id as a refernce
                   

                    # The user sees: The primary objective of this research is to improve...
                    st.subheader("🤖 AI Answer")
                    st.markdown(answer)

                    # User: What is the objective?
                    # Assistant: The objective is...
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

            except Exception as e:
                print(e)
                st.error("❌ Something went wrong while processing your request.")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "❌ Something went wrong while processing your request."
                    }
                )

                st.divider()
                st.caption("© 2026 AI Research Assistant | Built with Streamlit • FAISS • Gemini")

