import numpy as np
import os, sys
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn import preprocessing

from sonata.util import load_embed
import sonata
from scipy.sparse import csr_matrix
import scipy.sparse as sp 
import networkx as nx
import matplotlib.pyplot as plt

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
    print(n_nodes, np.max(edges), np.min(edges), data.toarray())
    dist = sp.csgraph.floyd_warshall(data, directed=False)
    print(dist, np.min(dist), np.max(dist))
    
    # if graph is not connected (usa-flights dataset)
    dist_max = np.nanmax(dist[dist != np.inf])
    dist[dist > dist_max] = 2*dist_max
    geo_mat = (dist - np.min(dist)) / (np.max(dist) - np.min(dist))

    # l1 distance
    sn_instance = sonata.model.sonata(0.1)
    l1_mat = sn_instance.geo_similarity(geo_mat)
    return l1_mat

def read_network(filename, undirected=True):
    if not undirected:
        return nx.read_edgelist(filename, nodetype=int, create_using=nx.DiGraph())
    return nx.read_edgelist(filename, nodetype=int)


def plt_acc(url, dataset_name, save_url):
    acc = np.loadtxt(url, dtype=str)
    methods = acc[:,0]
    acc = acc[:,1:]
    acc = acc.astype(float)
    print(acc, methods)

    plt.figure()
    x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    c= ['r', 'skyblue', 'limegreen', 'y', 'm', 'c', 'k', 'darkorange', 'mediumorchid']

    for i, m in enumerate(methods):
        plt.plot(x, acc[i], marker='o', linestyle='-', label=m, color=c[i])
    
    plt.title(f'{dataset_name} Node Classification')
    plt.xlabel('Percentage')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(save_url)    

def main(dataset_name, url, save_path):
    # methods = ['deepwalk', 'line', 'node2vec', 'struc2vec', 'eni_1_1']
    methods = ['sonata', 'drne', 'netmf', 'struc2vec']
    centrality_path = f'../result/graphs/centrality/{dataset_name}'
    if dataset_name.endswith('flights'):
        Y = np.loadtxt(f'../data/graphs/labels_{dataset_name}.txt').astype(int)
        labels = Y[:, 1]
    else:
        num_class = 4
        ground_truth = 'spread_number'
        Y = load_centrality(centrality_path, ground_truth)
        rY = np.sort(Y[:, 1])
        threshold = [rY[int(i*len(rY)/num_class)] for i in range(num_class)]+[rY[-1]+1]
        labels = np.array([len(list(filter(lambda x: i<x, threshold)))-1 for i in Y[:, 1]])
    lb = preprocessing.LabelBinarizer()
    labels = lb.fit_transform(labels)

    ### load embeddings
    # embedding_filenames = [os.path.join(save_path, "baseline_{}".format(embedding_size), 
    #     "{}.embeddings".format(m)) for m in methods if not m.startswith('eni_')]+\
    #     [os.path.join(save_path, "{}_{}".format(m, embedding_size), 'embeddings.npy') for m in methods if m.startswith('eni_')]
    # embedding_filenames = [os.path.join(save_path, "baseline_{}".format(embedding_size), "{}.npy".format(m)) for m in methods]
    # embeddings = [load_embeddings(name)[Y[:, 0].astype(int)] for name in embedding_filenames]

    edges = sonata.util.load_data(f"{url}.edgelist")
    edges = edges.astype(np.int32)
    embeddings = [get_sonata_l1(edges)[Y[:, 0].astype(int)]]  
    embeddings.append(load_embed(f"{save_path}/drne.npy")[Y[:, 0].astype(int)])  
    embeddings.append(load_embed(f"{save_path}/netmf.npy")[Y[:, 0].astype(int)])  
    embeddings.append(load_embed(f"{save_path}/struc2vec.emb")[Y[:, 0].astype(int)])  

    # load network & calculate centrality
    centrality_types = ['closeness', 'betweenness', 'eigenvector', 'kcore']
    G = read_network(f"{url}.edgelist", undirected=True)
    G.remove_edges_from(nx.selfloop_edges(G))
    is_connected = nx.is_connected(G)
    print("Is the graph connected?", is_connected)
    for c in centrality_types:
        centrality = get_centrality(G, c)
        save_centrality(centrality, centrality_path, c)

    # load centrality
    centralities = [load_centrality(centrality_path, c)[Y[:, 0].astype(int), 1].reshape(-1, 1) for c in centrality_types]
    for c in centralities:
        c = c.reshape(-1, 1)
    #res = np.zeros((len(methods), len(centrality_types)))
    combine_centrality = np.hstack(centralities)
    centralities.append(combine_centrality)

    print([i for i in methods]+[i for i in centrality_types])
    all_acc = []
    for radio in np.arange(0.1, 1.0, 0.1):
        acc = []
        for _ in range(100):
            index = np.random.permutation(range(len(labels)))
            th = int(radio*len(labels))
            temp_res = []
            for i in range(len(methods)):
                #lr = OneVsRestClassifier(svm.SVC(kernel='linear', C=0.025, probability=True))
                data = embeddings[i]
                lr = OneVsRestClassifier(LogisticRegression())
                lr.fit(data[index[:th]], labels[index[:th]])
                y_pred = lr.predict_proba(data[index[th:]])
                y_pred = lb.transform(np.argmax(y_pred, 1))
                #y_pred = lr.predict(embeddings[i][index[th:]])
                temp_res.append(np.sum(np.argmax(y_pred, 1) == np.argmax(labels[index[th:]], 1))/len(y_pred))
                #print(np.sum(y_pred == labels[index[th:]])/len(y_pred))
            for i in range(len(centralities)):
                data = centralities[i]
                lr = OneVsRestClassifier(LogisticRegression())
                lr.fit(data[index[:th]], labels[index[:th]])
                y_pred = lr.predict_proba(data[index[th:]])
                y_pred = lb.transform(np.argmax(y_pred, 1))
                #y_pred = lr.predict(embeddings[i][index[th:]])
                temp_res.append(np.sum(np.argmax(y_pred, 1) == np.argmax(labels[index[th:]], 1))/len(y_pred))
                #print(np.sum(y_pred == labels[index[th:]])/len(y_pred))
            acc.append(temp_res)
        acc = np.array(acc)
        #print(acc)
        print("radio={}, acc={}".format(radio, np.mean(acc, 0)))
        all_acc.append(np.mean(acc, 0))

    all_acc = np.array(all_acc)
    print(methods+centrality_types+["combined"])
    print(all_acc)
    arr_with_header = np.vstack((methods+centrality_types+["combined"], all_acc))
    print(arr_with_header.transpose())
    np.savetxt(f"{save_path}/classification.txt", arr_with_header.transpose(), fmt='%s', delimiter='\t')

if __name__ == '__main__':
    dataset_name = sys.argv[1]
    url=f"../data/graphs/{dataset_name}"
    save_path = '../result/graphs/{}'.format(dataset_name)

    main(dataset_name, url, save_path)

    plt_acc(url=f"{save_path}/classification_from_paper.txt", dataset_name=dataset_name, save_url=f"{save_path}/classification_from_paper.png")





    
