import streamlit as st 
import numpy as np
import shutil
import os 

from utils.pdf_loader import extract_text_from_Pdfs
from utils.text_splitter import split_text
from utils.embeddings import load_embedding_model
from utils.vector_stores import create_faiss_index, save_vector_store, load_vector_store
from utils.rag_chain import get_llm, generate_answer, generate_summary

st.set_page_config(page_title="AI Research Assistant", layout="wide")


st.title("📚 AI Research Assistant")

index, chunks = load_vector_store()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("AI Research Assistant")
    st.write("Powered by Gemini + FAISS")

    # Vector database status
    if index is not None:
        st.success("Vector DB Loaded")
    else:
        st.warning("No Saved Vector DB")    

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    # Delete saved FAISS database
    if st.button("Delete Knowledge Base"):
        if os.path.exists("vector_db"):
            shutil.rmtree("vector_db")
        st.rerun()



st.write("Uploads PDFs and ask questions")

if index is not None:
    st.success("Existing Knowledge Base Loaded")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if uploaded_files:
    with st.spinner("Processing PDFs..."):

        raw_text = extract_text_from_Pdfs(uploaded_files)

        chunks = split_text(raw_text)

        embedding_model = load_embedding_model()

        index, embeddings = (create_faiss_index(chunks, embedding_model))

        save_vector_store(index, chunks)

        llm = get_llm()

    st.success("Documents Processed")

    if st.button("Generate Document Summary..."):

        with st.spinner("Generating Summary..."):

            summary = generate_summary(llm, raw_text[:30000])

        st.subheader("Document Summary")

        st.write(summary)

        st.download_button(label = "Download Summary", data=summary, file_name="summary.txt", mine="text/plain")

    question = st.chat_input("Ask anything about the document...")

    if question: 

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        question_embedding = (embedding_model.encode([question]))

        question_embedding = (np.array(question_embedding).astype("float32"))

        k = min(5, len(chunks))

        # Search vector database
        # distances = similarity score
        # indices = chunk IDs
        distances, indices = (index.search(question_embedding, k))

        st.write("Indices:", indices)
        st.write("Number of Chunks:", len(chunks))

        # Store complete source information
        retrieved_chunks = []

        # Store complete source information
        retrieved_sources = []

        # Loop through chunk indexes returned by FAISS
        for idx in indices[0]:

            # Prevent index errors
            if idx < len(chunks):

                # Store chunk text
                retrieved_sources.append(
                    {
                        "chunk_id": idx,

                        "distance": float(
                            distances[0][
                                list(indices[0]).index(idx)
                            ]
                        ),
                        "text": chunks[idx]
                    }
                )
        
        # Create context for Gemini
        # Only text is sent to Gemini
        context = "\n\n".join(retrieved_chunks)

        st.write("Context Length:", len(context))

        print("=" * 50)
        print("CONTEXT LENGTH:", len(context))
        print(context[:1000])
        print("=" * 50)
        answer = generate_answer(llm, context, question, st.session_state.messages)

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        # Ex: What is lstm? and next question is Ex: what are its limitation then gemini know it's = LSTM 
        if len(st.session_state.messages) > 20:
            st.session_state.messages = (
                st.session_state.messages[-20:]
            )

        #Display sources used by Gemini
        st.subheader("Sources Used")


        for i, chunk in enumerate(retrieved_chunks, start=1):
            with st.expander(f"Sources {i}"):
                st.write(chunk)