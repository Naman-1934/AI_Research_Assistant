from sentence_transformers import SentenceTransformer


# Embedding is nothing but a Vector Representation of a particular Word. Ex: Artificial Intelligence, Vector: [0.12,0.45,0.88,...]
def load_embedding_model():

    # all-MiniLM-L6-v2 is Fast, Free, Good semantic search and Widely used in RAG systems.
    model = SentenceTransformer("all-MiniLM-L6-v2")

    return model