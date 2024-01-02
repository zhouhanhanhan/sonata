import numpy as np
import scanpy as sc
from sklearn.preprocessing import StandardScaler
import warnings
import sonata
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import os
warnings.filterwarnings("ignore")

def vis(data, label, mode="pca", show=True, path=""):
    if mode == "pca":
        pca = PCA(n_components=2).fit(data)
        data_embed = pca.fit_transform(data)
    elif mode == "umap":
        umaps = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                  metric="correlation").fit(data) # by default metric="euclidean", but diagonal align paper uses correlation
        data_embed = umaps.transform(data)
    elif mode == "tsne":
        data_embed = TSNE(n_components=2).fit_transform(data)
    fig = plt.figure()   
    plt.scatter(data_embed[:,0], data_embed[:,1], c = label, s=1, alpha=0.8) 
    if show:
        plt.show()
    else:
        plt.savefig(path)
        plt.close()

def plt_ambiguous_groups(
    data_mat: np.ndarray,
    ambiguous_cell_groups: dict,
    save_url: str = '',
    alpha: float = 0.8,
    show: bool = True,
    mode: str = "pca"
) -> None:
    if mode == "pca":
        pca = PCA(n_components=2).fit(data_mat)
        data_embed = pca.fit_transform(data_mat)
    elif mode == "umap":
        umaps = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                  metric="correlation").fit(data_mat) # metric="euclidean"?
        data_embed = umaps.transform(data_mat)
    elif mode == "tsne":
        data_embed = TSNE(n_components=2).fit_transform(data_mat)
    
    ambiguous_nodes = np.asarray([], dtype=int)
    ambiguous_labels = np.asarray([], dtype=int)
    for label, nodes in ambiguous_cell_groups.items():
        ambiguous_nodes = np.concatenate([ambiguous_nodes, nodes])
        ambiguous_labels = np.concatenate([ambiguous_labels, np.asarray([label]*len(nodes))])
    assert len(ambiguous_nodes) == len(ambiguous_labels)
    unambuguous_nodes = np.setdiff1d(list(range(data_mat.shape[0])), ambiguous_nodes)

    data_pca_ambiguous = data_embed[ambiguous_nodes, :]

    fig = plt.figure() 
    plt.scatter(list(data_embed[unambuguous_nodes, 0]), list(data_embed[unambuguous_nodes, 1]), 
                c="grey", alpha=alpha, label='Certain', s=1)

    if len(ambiguous_cell_groups) > 1:    
        for idx, class_label in enumerate(np.unique(ambiguous_labels)):
            class_indices = np.where(ambiguous_labels == class_label)[0]
            plt.scatter(list(data_pca_ambiguous[class_indices, 0]), list(data_pca_ambiguous[class_indices, 1]), alpha=alpha,
                        label='Ambiguous Group={}'.format(class_label), alpha=0.8, s=1)
        
    plt.legend()
    plt.title("Ambiguous groups")
    if show:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_url), exist_ok=True)
        plt.savefig(save_url, bbox_inches="tight")
        plt.close()


def dataset(url):
    ##Scenario 1: Same cellular *composition*, different features------------------
    adata = sc.read_h5ad(url)
    X = adata.X
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = adata.obs['cell'].values.tolist()
    s1_y = np.array(y)
    s1_unique_labels, s1_numeric_labels = np.unique(s1_y, return_inverse=True)

    s1_domain1 = X[:,:1200]
    s1_domain2 = X[:,800:]

    ##Scenario 2: One missing cell-type in each domain-----------------------------
    s2_domain1 = np.copy(s1_domain1)
    s2_domain2 = np.copy(s1_domain2)
    s2_y = np.copy(s1_y)
    ##deteling one cell type in each domain
    s2_domain1 = s2_domain1[y!="astrocytes_ependymal",:]
    s2_y1 = s2_y[s2_y!="astrocytes_ependymal"]
    s2_domain2 = s2_domain2[y!="endothelial-mural",:]
    s2_y2 = s2_y[s2_y!="endothelial-mural"]
    s2_unique_label1, s2_numeric_label1 = np.unique(s2_y1, return_inverse=True)
    s2_unique_label2, s2_numeric_label2 = np.unique(s2_y2, return_inverse=True)
    
    dataset = {
        "s1": {"domain1": s1_domain1, "domain2": s1_domain2, 
                "label1": s1_numeric_labels, "label2": s1_numeric_labels, 
                "unique_label1": s1_unique_labels, "unique_label2": s1_unique_labels},
        "s2": {"domain1": s2_domain1, "domain2": s2_domain2, 
                "label1": s2_numeric_label1, "label2": s2_numeric_label2,
                "unique_label1": s2_unique_label1, "unique_label2": s2_unique_label2}
    }
    return dataset

if __name__=="__main__":
    apx="../data/diagonal_align"
    h4c_file = "Zeisel.h5"
    save_url = "../result/diagonal/sonata"
    os.makedirs(save_url, exist_ok=True)

    datasets = dataset(os.path.join(apx, h4c_file))
    for scenario, param in datasets.items():
        print(f"Run scenario {scenario} ......")
        os.makedirs(os.path.join(save_url, scenario), exist_ok=True)
        # visualize domain data
        vis(param["domain2"], param["label2"], mode="pca", show=False, 
            path=os.path.join(save_url, scenario, f"pca_{scenario}_domain2.png"))
        vis(param["domain2"], param["label2"], mode="umap", show=False, 
            path=os.path.join(save_url, scenario, f"umap_{scenario}_domain2.png"))
        vis(param["domain2"], param["label2"], mode="tsne", show=False, 
            path=os.path.join(save_url, scenario, f"tsne_{scenario}_domain2.png"))

        # ambiguous sonata
        for sig in [0.1, 0.13, 0.15, 0.17, 0.2]:
            print(f"Run scenario {scenario} sigma = {sig} ......")
            sn_instance = sonata.model.sonata(sigma=sig)
            ambiguous_groups = sn_instance.check_ambiguity(param["domain2"])
            # visualize ambiguity groups
            plt_ambiguous_groups(param["domain2"], ambiguous_groups, show=False, 
                                        save_url=os.path.join(save_url, scenario,
                                        f"umap_{scenario}_sig{sig}_ambiguous_domain2.png"), mode="umap")