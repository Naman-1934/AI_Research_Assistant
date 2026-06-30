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
@st.cache_resource
def load_llm():
    return get_llm()

@st.cache_resource
def load_embeddings():
    return load_embedding_model()

llm = load_llm()
embedding_model = load_embeddings()


# ──────────────────────────────────────────────
# LOAD SAVED VECTOR DB FROM DISK
# If user already processed PDFs in a previous session,
# load that saved index and chunks automatically.
# Returns (None, None) if no saved DB exists yet.
# ──────────────────────────────────────────────
index = None
chunks = []

# ──────────────────────────────────────────────
# SESSION STATE — chat history
# st.session_state persists across reruns in the same session.
# Without this, chat history would reset every time the user types.
# ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "uploader_reset" not in st.session_state:
    st.session_state.uploader_reset = 0

# ──────────────────────────────────────────────
# SIDEBAR
# New Chat
# ──────────────────────────────────────────────
with st.sidebar:
    if st.button("New Chat"):

        # Clear chat 
        st.session_state.messages = []

        # Delete saved FAISS database
        if os.path.exists("vector_db"):
            shutil.rmtree("vector_db")

        st.session_state.vector_store = None

        st.session_state.faiss_index = None

        st.session_state.chunks = []

        st.session_state.uploader_reset += 1

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
    st.session_state.faiss_index = None
    st.session_state.chunks = []
    st.session_state.uploader_reset += 1

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
if "processed" not in st.session_state:
    st.session_state.processed = False
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

uploaded_file = st.file_uploader(
    "📄 Upload Research Paper (PDF)", 
    type="pdf",
    accept_multiple_files=False
    )


# ──────────────────────────────────────────────
# PDF PROCESSING
# Only runs when user actually uploads files.
# All 4 pipeline steps happen here:
# extract → chunk → embed → save to disk
# ──────────────────────────────────────────────
if uploaded_file is not None:

    st.session_state.processed = False
    st.session_state.processed_file = None

     # We check session_state to ensure we only process the file once.
    if uploaded_file and not st.session_state.processed != uploaded_file.name:

        with st.spinner("📄 Processing research paper..."):
            try:

                # Step 1 — extract raw text from all uploaded PDFs
                raw_text = extract_text_from_Pdfs(uploaded_file)

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
                faiss_index, embeddings = create_faiss_index(chunks, embedding_model)

                # Step 4 — save index + chunks to disk so they survive app restarts
                save_vector_store(faiss_index, chunks)

                st.session_state["faiss_index"] = faiss_index
                st.session_state["chunks"] = chunks
                st.session_state["vector_store"] = faiss_index
                st.session_state["raw_text_for_summary"] = raw_text

                st.session_state.raw_text_for_summary = raw_text

                # Mark this specific file as processed and save text for summary
                st.session_state["processed"] = True
                st.session_state["processed_file_name"] = uploaded_file.name

                st.success("✅ Research paper processed successfully.")

            except Exception as e:
                st.error("❌ An error occurred during processing:")
                st.exception(e)
    
  



    # ── SUMMARY SECTION ──────────────────────────────────────
    # Kept strictly inside `if uploaded_files:` because raw_text
    # only exists here. Moving it outside would cause NameError.
    # ─────────────────────────────────────────────────────────
    # 4. Display Status and Actions only after processing succeeds
    if st.session_state.processed:
        st.success(f"✅ Active Document: {st.session_state.processed_file}")
    if st.button("📝 Generate Document Summary"):
        if st.spinner("Generating summary..."):

            with st.spinner("📝 Generate Research Paper Summary"):
                # First 30,000 chars only — keeps us inside Gemini token limits
                st.write("Here is the summary of the paper...")

                raw_text  = st.session_state.get("raw_text_for_summary")

                if raw_text:
                    summary = generate_summary(llm, st.session_state.raw_text_for_summary[:85000])
                else:
                    st.error("❌ Document text not found. Please try re-uploading your PDF to process it.")

            st.subheader("📑 Research Paper Summary")
            st.write(summary)

            # FIX Bug 4: was mine="text/plain" — typo, silent failure
            # Corrected to mime="text/plain"
            st.download_button(label=" Download Summary", data=summary, file_name="summary.txt", mime="text/plain")
        else:
            st.error("Please upload and process a document first.")

# ──────────────────────────────────────────────
# Process User Question
# This creates the chatbot input field at the bottom.
# ──────────────────────────────────────────────
user_question = st.chat_input("💬 Ask anything about from the  uploaded research paper...")

if user_question:
    if st.session_state.vector_store is None:
        st.warning("Please upload and process a research paper first.")
    elif len(user_question.strip()) == 0:
        st.warning("Please enter a valid question.")
    else:

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
                    faiss_index = st.session_state.faiss_index
                    chunks = st.session_state.chunks
                    relevant_results = retrieve_relevant_chunks(faiss_index, chunks, user_question, top_k=5)

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

                        answer = generate_answer(llm=llm, question=user_question, context=context, chat_history=st.session_state.messages)

                        st.subheader("🤖 AI Answer")
                        st.markdown(answer)

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

