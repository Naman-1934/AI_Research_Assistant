# FAISS (Facebook AI Similarity Search) is an open-source library built by Meta that allows computers to quickly search, index, 
# and cluster massive collections of numerical vectors

import faiss
import numpy as np

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