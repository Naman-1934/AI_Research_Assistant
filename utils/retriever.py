from utils.embeddings import load_embedding_model
import numpy as np


def retrieve_relevant_chunks(index, chunks, question, top_k=3):
    """
    Retrieve the most relevant chunks from FAISS
    based on the user's question.
    """

    try:

        embedding_model = load_embedding_model()

        question_embedding = embedding_model.encode([question])

        question_embedding = np.array(question_embedding).astype("float32")

        distances, indices = index.search(
            question_embedding,
            top_k
        )

        results = []

        for i, idx in enumerate(indices[0]):

            if idx == -1:
                continue

            results.append(
                {
                    "content": chunks[idx],
                    "chunk_id": int(idx),
                    "score": float(distances[0][i])
                }
            )

        return results

    except Exception as e:

        raise Exception(
            f"Error retrieving chunks: {str(e)}"
        )