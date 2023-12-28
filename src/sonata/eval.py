from unioncom import UnionCom
import numpy as np
import sonata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from matplotlib import pyplot as plt
import os
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier


def transfer_accuracy(domain1, domain2, type1, type2):
	knn = KNeighborsClassifier()
	knn.fit(domain2, type2)
	type1_predict = knn.predict(domain1)
	count = 0
	for label1, label2 in zip(type1_predict, type1):
		if label1 == label2:
			count += 1
	return count / len(type1)

def test_labelTA(integrated_data, datatype):
	for i in range(len(integrated_data)-1):
		acc = transfer_accuracy(integrated_data[i], integrated_data[-1], datatype[i], datatype[-1])
		print("label transfer accuracy of data{:d}:".format(i+1))
		print(acc)    
	return acc

def calc_frac_idx(x1_mat,x2_mat):
	"""
	Returns fraction closer than true match for each sample (as an array)
	"""
	fracs = []
	x = []
	nsamp = x1_mat.shape[0]
	rank=0
	for row_idx in range(nsamp):
		euc_dist = np.sqrt(np.sum(np.square(np.subtract(x1_mat[row_idx,:], x2_mat)), axis=1))
		true_nbr = euc_dist[row_idx]
		sort_euc_dist = sorted(euc_dist)
		rank =sort_euc_dist.index(true_nbr)
		frac = float(rank)/(nsamp -1)

		fracs.append(frac)
		x.append(row_idx+1)

	return fracs,x

def calc_domainAveraged_FOSCTTM(x1_mat, x2_mat, links):
	"""
	Outputs average FOSCTTM measure (averaged over both domains)
	Get the fraction matched for all data points in both directions
	Averages the fractions in both directions for each data point
	"""
	sorted_x1_mat = x1_mat[np.transpose(links)[0]]
	sorted_x2_mat = x2_mat[np.transpose(links)[1]]

	fracs1,xs = calc_frac_idx(sorted_x1_mat, sorted_x2_mat)
	fracs2,xs = calc_frac_idx(sorted_x2_mat, sorted_x1_mat)
	fracs = []
	for i in range(len(fracs1)):
		fracs.append((fracs1[i]+fracs2[i])/2)  
	return fracs


def visualize(data, data_integrated, datatype=None, mode='PCA', show=False, filename="fig.png"):
    assert (mode in ["PCA", "UMAP", 'TSNE']), "mode has to be either one of 'PCA', 'UMAP', or 'TSNE'."
    dataset_num = len(data)

    data_all = np.vstack((data_integrated[0], data_integrated[1]))
    for i in range(2, dataset_num):
        data_all = np.vstack((data_all, data_integrated[i]))
    if mode=='PCA':
        embedding_all = PCA(n_components=2).fit_transform(data_all)
    elif mode=='TSNE':
        embedding_all = TSNE(n_components=2).fit_transform(data_all)
    else:
        embedding_all = umap.UMAP(n_components=2).fit_transform(data_all)

    tmp = 0
    num = [0]
    for i in range(dataset_num):
        num.append(tmp+np.shape(data_integrated[i])[0])
        tmp += np.shape(data_integrated[i])[0]

    embedding = []
    for i in range(dataset_num):
        embedding.append(embedding_all[num[i]:num[i+1]])

    color = [[1,0.5,0], [0.2,0.4,0.1], [0.1,0.2,0.8], [0.5, 1, 0.5], [0.1, 0.8, 0.2]]
    
    fig = plt.figure(figsize=(10, 5))
    if datatype is not None:
        datatype_all = np.hstack((datatype[0], datatype[1]))
        for i in range(2, dataset_num):
            datatype_all = np.hstack((datatype_all, datatype[i]))

        plt.subplot(1,2,1)
        for i in range(dataset_num):
            plt.scatter(embedding[i][:,0], embedding[i][:,1], c=color[i], s=5., alpha=0.8)
        plt.title('Integrated Embeddings')
        if mode=='PCA':
            plt.xlabel('PCA-1')
            plt.ylabel('PCA-2')
        elif mode=='TSNE':
            plt.xlabel('TSNE-1')
            plt.ylabel('TSNE-2')
        else:
            plt.xlabel('UMAP-1')
            plt.ylabel('UMAP-2')

        plt.subplot(1,2,2)
        for j in set(datatype_all):
            index = np.where(datatype_all==j)  
            plt.scatter(embedding_all[index,0], embedding_all[index,1], s=5., alpha=0.8)
            
        plt.title('Integrated Cell Types')
        if mode=='PCA':
            plt.xlabel('PCA-1')
            plt.ylabel('PCA-2')
        elif mode=='TSNE':
            plt.xlabel('TSNE-1')
            plt.ylabel('TSNE-2')
        else:
            plt.xlabel('UMAP-1')
            plt.ylabel('UMAP-2')
    else:
        for i in range(dataset_num):
            plt.scatter(embedding[i][:,0], embedding[i][:,1], c=color[i], s=5., alpha=0.8)
        plt.title('Integrated Embeddings')
        if mode=='PCA':
            plt.xlabel('PCA-1')
            plt.ylabel('PCA-2')
        elif mode=='TSNE':
            plt.xlabel('TSNE-1')
            plt.ylabel('TSNE-2')
        else:
            plt.xlabel('UMAP-1')
            plt.ylabel('UMAP-2')
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(filename)
    
def plt_acc(label, acc, title="", x_label="", y_label="", save_path=""):
    plt.figure()
    plt.plot(label, acc)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(save_path)