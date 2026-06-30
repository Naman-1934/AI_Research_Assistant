from utils.embeddings import load_embedding_model
import numpy as np


def retrieve_relevant_chunks(index, chunks, question, top_k=3):
    """
    Retrieve the most relevant chunks from FAISS
    based on the user's question.
    """

    try:
        embedding_model = load_embedding_model()

        # This line of code will converts sentence into a vector:
        question_embedding = embedding_model.encode([question])

        question_embedding = np.array(question_embedding).astype("float32")

    # FAISS performs similarity search. Ex: FAISS finds: Chunk 8, Chunk 15 and Chunk 22, because those chunks are semantically closest.
        distances, indices = index.search(question_embedding, top_k)

        results = []

        # FAISS returns positions.Ex: [8, 15, 22], We convert those positions back into actual text chunks.
        for idx in indices[0]:
            if idx != -1:
                results.append(
                    "content": chunks[idx],
                    "chunk_id": int(idx),
                    :"score": float(distances[0][1])
                    )
        return results
    
    except Exception as e:
        raise Exception(f"Error retrieving chunks: {str(e)}")