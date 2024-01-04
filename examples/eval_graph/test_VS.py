import numpy as np
import os, sys
import networkx as nx
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import sonata
from scipy.sparse import csr_matrix
import scipy.sparse as sp 
import scipy

def read_network(filename, undirected=True):
    if not undirected:
        return nx.read_edgelist(filename, nodetype=int, create_using=nx.DiGraph())
    return nx.read_edgelist(filename, nodetype=int)

def VS(A, alpha, iter_num=100):
    """the implement of Vertex similarity in networks"""
    assert 0 < alpha < 1
    print(type(A))
    assert type(A) is scipy.sparse.csr.csr_matrix
    lambda_1 = scipy.sparse.linalg.eigsh(A, k=1, which='LM', return_eigenvectors=False)[0]
    n = A.shape[0]
    d = np.array(A.sum(1)).flatten()
    d_inv = np.diag(1./d)
    dsd = np.random.normal(0, 1/np.sqrt(n), (n, n))
    #dsd = np.zeros((n, n))
    I = np.eye(n)
    for i in range(iter_num):
        dsd = alpha/lambda_1*A.dot(dsd)+I
        if i % 10 == 0:
            print('VS', i, '/', iter_num)
    return d_inv.dot(dsd).dot(d_inv)

def compare(S1, S2):
    S1 /= np.sum(S1)
    S2 /= np.sum(S2)
    return np.mean((S1-S2)**2)

def resort(rank):
    res = np.zeros_like(rank)
    for i, j in enumerate(rank):
        res[j] = i
    return res

def similarity(x, y):
    return stats.kendalltau(x, y)[0]

def get_sonata_l1(edges):
    # construct knn(k=real connected nodes)
    n_nodes = len(np.unique(edges))
    data = csr_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])), shape=(n_nodes, n_nodes))
    dist = sp.csgraph.floyd_warshall(data, directed=False)
    geo_mat = (dist - np.min(dist)) / (np.max(dist) - np.min(dist))

    # l1 distance
    sn_instance = sonata.model.sonata(0.1)
    l1_mat = sn_instance.geo_similarity(geo_mat)
    return l1_mat

if __name__ == '__main__':
    dataset_name = 'jazz'
    url = f"../data/graphs/{dataset_name}"
    methods = ['sonata', 'drne']

    save_path = '../result/graphs/{}'.format(dataset_name)
    edges = sonata.util.load_data(f"{url}.edgelist")
    edges = edges.astype(np.int32)
    embeddings = [get_sonata_l1(edges)]
    embeddings.append(sonata.util.load_embed(f"{save_path}/baselines/embeddings.npy"))
    print(embeddings[0].shape)
    print(embeddings[1].shape)
  
    G = read_network(f"{url}.edgelist")
    A = nx.to_scipy_sparse_matrix(G, dtype=float)
    # A = nx.to_scipy_sparse_array(G, dtype=float)
    S = VS(A, 0.9, 100)
    S = (S+S.T)/2
    aS = np.argsort(S, None)
    arS = resort(aS)
    for i, e in enumerate(embeddings):
        if methods[i].startswith('eni') or methods[i] == 'struc2vec':
            E = -squareform(pdist(e, 'euclidean'))
        else:
            E = e.dot(e.T)
        aE = np.argsort(E, None)
        arE = resort(aE)
        # print("VS rank: ", arS)
        # print("embed rank: ", arE)
        print(methods[i], similarity(arS, arE), sep='\t')
