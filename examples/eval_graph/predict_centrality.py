import numpy as np
import sys, os

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict

import sonata
from scipy.sparse import csr_matrix
import scipy.sparse as sp 
import networkx as nx

def MSE(x, y):
    return np.mean((x-y)**2)

def print_array(X):
    a, b = X.shape
    print("\n".join(["\t".join(["{:.6e}"]*b)]*a).format(*X.flatten()))


def load_centrality(file_path, type_):
    filename = os.path.join(file_path, "{}.index".format(type_))
    data = np.loadtxt(filename)
    if type_ == 'spread_number':
        return data
    return np.vstack((np.arange(len(data)), data)).T

def save_centrality(data, file_path, type_):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    print("\n".join(map(str, data.values())), file=open(os.path.join(file_path, "{}.index".format(type_)), 'w'))

def get_centrality(network, type_='degree', undirected=True):
    if type_ == 'degree':
        if undirected:
            return nx.degree_centrality(network)
        else:
            return nx.in_degree_centrality(network)
    elif type_ == 'closeness':
        return nx.closeness_centrality(network)
    elif type_ == 'betweenness':
        return nx.betweenness_centrality(network)
    elif type_ == 'eigenvector':
        return nx.eigenvector_centrality(network)
    elif type_ == 'kcore':
        return nx.core_number(network)
    else:
        return None
        
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

def read_network(filename, undirected=True):
    if not undirected:
        return nx.read_edgelist(filename, nodetype=int, create_using=nx.DiGraph())
    return nx.read_edgelist(filename, nodetype=int)

if __name__ == '__main__':
    dataset_name = sys.argv[1]
    url=f"../data/graphs/{dataset_name}"

    # load data
    edges = sonata.util.load_data(f"{url}.edgelist")
    edges = edges.astype(np.int32)
    # calculate l1_mat
    l1_mat = get_sonata_l1(edges)
    np.savetxt(f"../result/graphs/{dataset_name}/sonata_l1.txt", l1_mat)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=4).fit(l1_mat)
    data_embed = pca.fit_transform(l1_mat)
    embedding = data_embed

    # load network
    G = read_network(f"{url}.edgelist", undirected=True)
    G.remove_edges_from(nx.selfloop_edges(G))

    centrality_types = ['degree', 'closeness', 'betweenness', 'eigenvector', 'kcore']
    centrality_path = '../result/graphs/centrality/{}'.format(dataset_name)
    save_path = '../result/graphs/{}'.format(dataset_name)
    
    for c in centrality_types:
        centrality = get_centrality(G, c)
        save_centrality(centrality, centrality_path, c)


    centralities=[load_centrality(centrality_path, c) for c in centrality_types]
    res = np.zeros(len(centrality_types))
    for i in range(len(centrality_types)):
        lr = LinearRegression(n_jobs=-1)
        y_pred = cross_val_predict(lr, embedding[centralities[i][:, 0].astype(int)], centralities[i][:, 1])
        res[i] = MSE(y_pred, centralities[i][:, 1])/np.mean(centralities[i][:, 1])    
    # print_array(res)
    print(res)  
    # np.savetxt(os.path.join(save_path, 'centrality'), res)  
    

