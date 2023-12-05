import numpy as np
import faiss

def generate_data(d, nb, nq):
    np.random.seed(1234)
    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 1000.
    xq = np.random.random((nq, d)).astype('float32')
    xq[:, 0] += np.arange(nq) / 1000.
    return xb, xq

def build_and_search_index(xb, xq, d, k):
    index = faiss.IndexFlatL2(d)
    #print(index.is_trained)
    
    index.add(xb)
    #print(index.ntotal)
    
    D, I = index.search(xb[:5], k)
    #print(I)
    #print(D)
    
    D, I = index.search(xq, k)
    
    print(I[:5])                   
    print(I[-5:])    

# Parameters
d = 64
nb = 100000
nq = 10000
k = 3

# Generate data
xb, xq = generate_data(d, nb, nq)

# Build and search index
build_and_search_index(xb, xq, d, k)