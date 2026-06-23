import streamlit as st 
import numpy as np

from utils.pdf_loader import extract_text_from_Pdfs
from utils.text_splitter import split_text
from utils.embeddings import load_embedding_model
from utils.vector_stores import create_faiss_index
from utils.rag_chain import get_llm, generate_answer

st.set_page_config(page_title="AI Research Assistant", layout="wide")

if "message" not in st.session_start():
    st.session_state.messages = []

with st.sidebar:
    st.header("AI Research Assistant")
    st.write("Powered by Gemini + FAISS")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

st.title("📚 AI Research Assistant")

st.write("Uploads PDFs and ask questions")

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

        llm = get_llm()

    st.success("Documents Processed")

    st.sidebar.success(f"{len(chunks)} Chunks Created")

    st.sidebar.info(f"{len(raw_text)} Characters")

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