# FAISS (Facebook AI Similarity Search) is an open-source library built by Meta that allows computers to quickly search, index, 
# and cluster massive collections of numerical vectors

import faiss
import numpy as np
import os
import pickle

def create_faiss_index(chunks, model):

    # Converts all chunks to vectors.
    embeddings = model.encode(chunks)

    embeddings = np.array(embeddings).astype("float32")

    # Ex: (50,384) = 50 chunks, 384 dimensions and it returns 384.
    dimension = embeddings.shape[1]

    # IndexFlatL2 is uses uclidean distance to find nearest vectors.
    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index, embeddings

def save_vector_store(index, chunks):
    
    os.makedirs("vector_db", exist_ok=True)

    faiss.write_index(index, "vector_db/faiss.index")

    with open("vector_db/chunks.pickle", "wb") as f:
        pickle.dump(chunks, f)


def load_vector_store():

    index_path = (
        "vector_db/faiss.index"
    )

    chunks_path = (
        "vector_db/chunks.pkl"
    )

    if(
        not os.path.exists(index_path)
        or 
        not os.path.exists(chunks_path)
    ):
        
        return None, None
    
    index = faiss.read_index(index_path)

    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks