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


# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("📚 AI Research Assistant")

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
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("AI Research Assistant")
    # st.write("Powered by Gemini + FAISS")

    st.divider()


    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

        # Reset index and chunks so the rest of the app
        # Immediately knows there is no DB anymore
        index = None
        chunks = []
        st.rerun()

# ──────────────────────────────────────────────
# STATUS BANNER
# Tells the user clearly what state the app is in.
# ──────────────────────────────────────────────
if index is not None:
    st.info("Ask questions below or upload new PDFs to replace it.")
else:
    st.write("Upload your PDFs below to get started.")


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
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)




# ──────────────────────────────────────────────
# PDF PROCESSING
# Only runs when user actually uploads files.
# All 4 pipeline steps happen here:
# extract → chunk → embed → save to disk
# ──────────────────────────────────────────────
if uploaded_files:
    with st.spinner("Processing PDFs - extracting, chunking, embedding..."):

        # Step 1 — extract raw text from all uploaded PDFs
        raw_text = extract_text_from_Pdfs(uploaded_files)

        # Step 2 — split into 1000-char chunks with 200-char overlap
        chunks = split_text(raw_text)

        # Step 3 — embed all chunks and build FAISS index
        index, embeddings = create_faiss_index(chunks, embedding_model)

        # Step 4 — save index + chunks to disk so they survive app restarts
        save_vector_store(index, chunks)

    st.success(f"✅ Processed  {len(chunks)} chunks from {len(uploaded_files)} PDF(s)")

    # ── SUMMARY SECTION ──────────────────────────────────────
    # Kept strictly inside `if uploaded_files:` because raw_text
    # only exists here. Moving it outside would cause NameError.
    # ─────────────────────────────────────────────────────────
    if st.button("📝 Generate Document Summary"):
        with st.spinner("Generating Summary..."):
            # First 30,000 chars only — keeps us inside Gemini token limits
            summary = generate_summary(llm, raw_text[:30000])

        st.subheader("Document Summary")
        st.write(summary)

        # FIX Bug 4: was mine="text/plain" — typo, silent failure
        # Corrected to mime="text/plain"
        st.download_button(label="⬇️ Download Summary", data=summary, filename="summary.txt", mime="text/plain")


# ──────────────────────────────────────────────
# Process User Question
# This creates the chatbot input field at the bottom.
# ──────────────────────────────────────────────
user_question = st.chat_input("Ask a question about your documents...")

if user_question:

    # Save User Message
    st.session_state.message.append(
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
                relevant_results = retrieve_relevant_chunks(st.session_state.vector_store, user_question, top_k=3)

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
                    answer = generate_answer(question=user_question, context=context, chat_history=st.session_state.messages)

                    #  Where did you find this answer so, we will return chunk_id as a refernce
                    sources = "\n".join(
                        [
                            f"sources chunk {item['chunk_id']}"
                            for item in relevant_results
                        ]
                    )

                    answer += (
                        "\n\n --- \n\n"
                        "**Sources Used:**\n\n"
                        f"{sources}"
                    )

                    # The user sees: The primary objective of this research is to improve...
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
                error_message = (f"error: {str(e)}")

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message
                    }
                )

