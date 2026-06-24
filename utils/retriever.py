from utils.embeddings import load_embedding_model


def retrieve_relevant_chunks(vector_store, question, top_k=3):
    """
    Retrieve the most relevant chunks from FAISS
    based on the user's question.
    """

    try:
        embedding_model = load_embedding_model()

        # This line of code will converts sentence into a vector:
        question_embedding = embedding_model.encode([question])

    # FAISS performs similarity search. Ex: FAISS finds: Chunk 8, Chunk 15 and Chunk 22, because those chunks are semantically closest.
        distances, indices = vector_store.index.search(question_embedding, top_k)

        results = []

        # FAISS returns positions.Ex: [8, 15, 22], We convert those positions back into actual text chunks.
        for idx in indices[0]:
            if idx != -1:
                results.append(vector_store.chunks[idx])
        return results
    
    except Exception as e:
        raise Exception(f"Error retrieving chunks: {str(e)}")