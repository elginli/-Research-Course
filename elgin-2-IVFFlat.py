import numpy as np
import faiss

def generate_data(d, nb, nq):
    np.random.seed(1234)
    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 1000.
    xq = np.random.random((nq, d)).astype('float32')
    xq[:, 0] += np.arange(nq) / 1000.
    return xb, xq

def build_index(xb, d, nlist):
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    #faiss.METRIC_L2

    assert not index.is_trained
    index.train(xb)
    assert index.is_trained
    index.add(xb)

    return index
   
def search_index(index, xb, xq, k):
    D, I = index.search(xq, k)
    print(I[-5:])
    index.nprobe = 5
    D, I = index.search(xq, k)
    print(I[-5:])

# Parameters
d = 64
nb = 100000
nq = 10000
k = 3
nlist = 100

# Generate data
xb, xq = generate_data(d, nb, nq)

# Build index
index = build_index(xb, d, nlist)

#Search index
search_index(index, xb, xq, k)
