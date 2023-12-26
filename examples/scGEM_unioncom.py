from unioncom import UnionCom
import numpy as np
import sonata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from matplotlib import pyplot as plt
import os
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

def visualize(data, data_integrated, datatype=None, mode='PCA', show=False, filename="fig.png"):
    assert (mode in ["PCA", "UMAP", 'TSNE']), "mode has to be either one of 'PCA', 'UMAP', or 'TSNE'."
    dataset_num = len(data)

    # styles = ['g', 'r', 'b', 'y', 'k', 'm', 'c', 'greenyellow', 'lightcoral', 'teal'] 
    # data_map = ['Chromatin accessibility', 'DNA methylation', 'Gene expression']
    # color_map = ['E5.5','E6.5','E7.5']
    embedding = []
    dataset_xyz = []
    for i in range(dataset_num):
        dataset_xyz.append("data{:d}".format(i+1))
        if mode=='PCA':
            embedding.append(PCA(n_components=2).fit_transform(data[i]))
        elif mode=='TSNE':
            embedding.append(TSNE(n_components=2).fit_transform(data[i]))
        else:
            embedding.append(umap.UMAP(n_components=2).fit_transform(data[i]))
   
    fig = plt.figure()
    if datatype is not None:
        for i in range(dataset_num):
            plt.subplot(1,dataset_num,i+1)
            for j in set(datatype[i]):
                index = np.where(datatype[i]==j) 
                plt.scatter(embedding[i][index,0], embedding[i][index,1], s=5.)
            plt.title(dataset_xyz[i])
            if mode=='PCA':
                plt.xlabel('PCA-1')
                plt.ylabel('PCA-2')
            elif mode=='TSNE':
                plt.xlabel('TSNE-1')
                plt.ylabel('TSNE-2')
            else:
                plt.xlabel('UMAP-1')
                plt.ylabel('UMAP-2')
            # plt.title(data_map[i])
    else:
        for i in range(dataset_num):
            plt.subplot(1,dataset_num,i+1)
            plt.scatter(embedding[i][:,0], embedding[i][:,1], s=5.)
            plt.title(dataset_xyz[i])
            if mode=='PCA':
                plt.xlabel('PCA-1')
                plt.ylabel('PCA-2')
            elif mode=='TSNE':
                plt.xlabel('TSNE-1')
                plt.ylabel('TSNE-2')
            else:
                plt.xlabel('UMAP-1')
                plt.ylabel('UMAP-2')
            plt.title(dataset_xyz[i])
    plt.tight_layout()

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
    # marker=['x','^','o','*','v']
    
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

data_url1 = "../data/scGEM/scGEM_expression.csv"
data_url2 = "../data/scGEM/scGEM_methylation.csv"
label_url1 = "../data/scGEM/scGEM_typeExpression.txt"
label_url2 = "../data/scGEM/scGEM_typeMethylation.txt" 

data1 = sonata.util.load_data(data_url1)
data2 = sonata.util.load_data(data_url2)
type1 = np.loadtxt(label_url1)
type2 = np.loadtxt(label_url2)
type1 = type1.astype(np.int)
type2 = type2.astype(np.int)

basename_unioncom = "../result/scGEM/unioncom"
basename_scot = "../result/scGEM/scot"
os.makedirs(os.path.join(basename_unioncom, "align_figs"), exist_ok=True)
os.makedirs(os.path.join(basename_scot, "align_figs"), exist_ok=True)

acc_unioncom = []
acc_scot = []
data1 = sonata.util.wrapped_normalize(data1)
data2 = sonata.util.wrapped_normalize(data2)

# for i in range(1,11):
#     # run unioncom
#     uc = UnionCom.UnionCom()
#     integrated_data = uc.fit_transform(dataset=[data1,data2])
#     acc = test_labelTA(integrated_data, [type1,type2])
#     acc_unioncom.append(acc)
#     visualize([data1,data2], integrated_data, [type1,type2], mode='PCA', 
#     filename=os.path.join(basename_unioncom, "align_figs", "align_label{}.png".format(i)))

# with open(os.path.join(basename_unioncom, 'accLT_log.txt'), 'w') as f:
#     f.write(f'Epoch\tLTAcc\n')   
#     for i, acc in enumerate(acc_unioncom):
#         f.write(f'{i+1}\t{acc}\n')   


for i in range(1,11):
    # run scot
    scot_instance = sonata.scotv1.SCOT(data1, data2)
    X_aligned, y_aligned = scot_instance.align(k=25, e=0.005, metric="correlation", normalize=False)
    integrated_data = [X_aligned, y_aligned]
    acc = test_labelTA(integrated_data, [type1,type2])
    acc_scot.append(acc)
    visualize([data1,data2], integrated_data, [type1,type2], mode='PCA', 
    filename=os.path.join(basename_scot, "align_figs", "align_label{}.png".format(i)))

with open(os.path.join(basename_scot, 'accLT_log.txt'), 'w') as f:
    f.write(f'Epoch\tLTAcc\n')   
    for i, acc in enumerate(acc_scot):
        f.write(f'{i+1}\t{acc}\n')   