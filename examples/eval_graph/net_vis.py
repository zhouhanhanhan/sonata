import numpy as np 
import matplotlib.pyplot as plt 
import sonata
import scipy.sparse as sp 
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
from skbio.stats.ordination import pcoa
import networkx as nx

def vis(data, label=[], mode="pca", show=True, path=""):
    if mode == "pca":
        pca = PCA(n_components=2).fit(data)
        data_embed = pca.fit_transform(data)
    elif mode == "umap":
        umaps = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                  metric="correlation").fit(data) 
        data_embed = umaps.transform(data)
    elif mode == "tsne":
        data_embed = TSNE(n_components=2).fit_transform(data)
    else:
        data_embed = data
    fig = plt.figure()  

    if len(label) == 0: 
        # for barbell dataset
        color_map = np.array(['xkcd:royal blue', 'xkcd:greenish', 'xkcd:blood red', 'xkcd:dark sky blue', 'xkcd:yellow ochre', 'xkcd:purple', 'xkcd:light grey'])
        index = [0]*9+[1, 2, 3, 4, 5, 6, 6, 5 , 4, 3 ,2 ,1]+[0]*9
        label = color_map[index]
        m = ['.']*15+['*']*15

    # for all
    plt.scatter(data_embed[:,0], data_embed[:,1], c=label, cmap=plt.cm.get_cmap('RdYlBu'), s=100, alpha=0.8) 

    # plt.scatter(data_embed[:,0][:14], data_embed[:,1][:14], c=label[:14], marker="*", s=100, alpha=0.3) 
    # plt.scatter(data_embed[:,0][15:], data_embed[:,1][15:], c=label[15:], marker="^", s=100, alpha=0.3) 
    # for i in range(len(index)):
    #     plt.annotate(index[i], (data_embed[:,0][i], data_embed[:,1][i]))
    if show:
        plt.show()
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.close()


def vis_graph_structure(filename, color="", save_path="graph.png"):
    '''
    This function is to visualize:
    1. graph structure
    2. embedding similarity (applicable or not? how to achieve?)
    '''
    print("reading edges...")
    G = nx.read_edgelist(filename, nodetype=int, create_using=nx.Graph())
    print("drawing graph...")

    nx.draw(G, node_color=color, pos=nx.kamada_kawai_layout(G), node_size=200, 
        cmap=plt.cm.Blues, with_labels=False, edge_color='grey', width=0.1, alpha=0.8)
    
    # nx.draw_circular(G, with_labels=False)
    # nx.draw(G, pos=nx.spring_layout(G), with_labels=False, node_size=10, width=0.1) #, node_color='red', font_color='white'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)

def vis_graph_node(filename, pos="", save_path=""):

    print("reading edges...")
    G = nx.read_edgelist(filename, nodetype=int, create_using=nx.Graph())
    print("drawing graph...")

    c = ["r"]+["b"]*29
    nx.draw(G, node_color=c, pos=nx.kamada_kawai_layout(G), node_size=200, 
        cmap=plt.cm.Blues, with_labels=False, edge_color='grey', width=0.1, alpha=0.8)
    plt.savefig(save_path)
  

if __name__== "__main__":
    dataset = "karate"
    url = f"../data/graphs/{dataset}"
    save_path = f"../result/graphs/{dataset}"

    # load data
    edges = sonata.util.load_data(f"{url}.edgelist")
    edges = edges.astype(np.int32)
    # for karate visualize its centrality
    if dataset == "karate":
        centrality = np.loadtxt(f"{url}_kcore.index")
        label=centrality
    else:
        label=[]


    # construct knn(k=real connected nodes)
    n_nodes = len(np.unique(edges))
    data = csr_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])), shape=(n_nodes, n_nodes))
    dist = sp.csgraph.floyd_warshall(data, directed=False)
    geo_mat = (dist - np.min(dist)) / (np.max(dist) - np.min(dist))
    vis(np.sort(geo_mat, axis=1), label=label, show=False, path=os.path.join(save_path, "sonata_geo_sorted.png"))
    
    # l1 distance
    sn_instance = sonata.model.sonata(0.1)
    l1_mat = sn_instance.geo_similarity(geo_mat)
    vis(l1_mat, label=label, show=False, path=os.path.join(save_path, "sonata_l1.png"))

    # cell-wise ambiguity
    cell_amat = sn_instance.cell_ambiguity(geo_mat)
    print(cell_amat)
    vis(cell_amat, label=label, show=False, path=os.path.join(save_path, "sonata_cell_ambiguity.png"))
    
    # # PCoA
    # dist_amat = cdist(cell_amat, cell_amat, 'euclidean')
    # pcoa_results = pcoa(dist_amat)
    # vis(dist_amat, show=False, path="../result/graphs/barbell/sonata_cell_ambiguity_pcoa.png")

    # load baselines
    baselines = ["struc2vec.emb", "drne.npy", "drne_16.npy", "netmf.npy"] # "node2vec.emb", "barbell_s2v.emb", "drne_16.npy"
    for baseline in baselines:
        basename = baseline.split(".")[0]
        embed = sonata.uitl.load_embed(os.path.join(save_path, "baselines", baseline))
        if basename == "netmf" and dataset == "karate":
            embed = embed[1:,:]
        print(baseline, "embed shape=", embed.shape)
        print(embed)

        if basename in ["drne_16", "barbell_s2v"]:
            vis(embed, label=label, show=False, path=os.path.join(save_path, f"{basename}.png"))
        else:
            vis(embed, label=label, show=False, path=os.path.join(save_path, f"{basename}.png"), mode="")

        #cosine similarity
        cosine_mat = cdist(embed, embed, 'cosine')
        vis_graph_structure(f"{url}.edgelist", cosine_mat[0], os.path.join(save_path, "graph", f"{basename}_node0.png"))
    
    # visualize a node of barbell
    # vis_graph_node(f"{url}.edgelist", save_path=os.path.join(save_path, "graph",  "pos0.png"))
    vis_graph_structure(f"{url}.edgelist", l1_mat[0], os.path.join(save_path, "graph", "l1_node0.png"))
    vis_graph_structure(f"{url}.edgelist", cell_amat[0], os.path.join(save_path, "graph",  "ambiguity_node0.png"))



