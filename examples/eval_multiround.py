from unioncom import UnionCom
import numpy as np
import sonata
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import os
import sys
import random

def test_multiround_scot(n, k, e, metric, data1, data2, links, basename):
    print(f"Testing scot(k={k}) for {n} rounds......")
    dirname = f"multiround_k{k}_e{e}_{metric}"
    os.makedirs(os.path.join(basename, dirname, "align_figs"), exist_ok=True)

    scot_instance = sonata.scotv1.SCOT(data1, data2)
    X_aligned, y_aligned = scot_instance.align(k=k, e=e, metric=metric, normalize=False)
    integrated_data = [X_aligned, y_aligned]

    # visulize alignment
    sonata.eval.visualize([data1,data2], integrated_data, [type1,type2], mode='PCA', 
    filename=os.path.join(basename, dirname, "align_figs", f"align_label{n}.png"))

    # label transfer accuracy
    acc = sonata.eval.test_labelTA(integrated_data, [type1,type2])

    # FOSCTTM error
    fracs = sonata.eval.calc_domainAveraged_FOSCTTM(X_aligned, y_aligned, links)

    # save acc & FOSCTTM error
    acc_log_url = os.path.join(basename, dirname, 'accLT_log.txt')
    foscttm_log_url = os.path.join(basename, dirname, 'FOSCTTM_log.txt')
    if not os.path.exists(acc_log_url):
        with open(acc_log_url, 'w') as f:
            f.write(f'Round\tLTAcc\n')
    if not os.path.exists(foscttm_log_url):
        with open(foscttm_log_url, 'w') as f:
            f.write(f'Round\tAVG_FOSCTTM\n') 

    with open(os.path.join(basename, dirname, 'accLT_log.txt'), 'a') as f:
        f.write(f'{n+1}\t{acc}\n')  
    with open(os.path.join(basename, dirname, 'FOSCTTM_log.txt'), 'a') as f: 
        f.write(f'{n+1}\t{np.mean(fracs)}\n')  


def datasets():
    data_params = {
        # "scGEM": {"data_url1": "scGEM/scGEM_expression.csv", "data_url2": "scGEM/scGEM_methylation.csv", 
        #           "label_url1": "scGEM/scGEM_typeExpression.txt", "label_url2": "scGEM/scGEM_typeMethylation.txt", 
        #           "scot_k": 25, "scot_e": 5e-3, "smetric": "correlation",
        #           "sigma": 0.15, "norm": "l2"},
        # "SNARE-seq": {"data_url1": "SNARE-seq/scrna_feat.npy", "data_url2": "SNARE-seq/scatac_feat.npy", 
        #           "label_url1": "SNARE-seq/SNAREseq_rna_types.txt", "label_url2": "SNARE-seq/SNAREseq_atac_types.txt", 
        #           "scot_k": 110, "scot_e": 1e-3, "smetric": "correlation",
        #           "sigma": 0.1, "norm": "l2"},
        "t_branch": {"data_url1": "t_branch/domain1.txt", "data_url2": "t_branch/domain2.txt", 
                  "label_url1": "t_branch/label_domain1.txt", "label_url2": "t_branch/label_domain2.txt", 
                  "scot_k": 10, "scot_e": 0.001, "smetric": "euclidean",
                  "sigma": 0.1, "curve_style": True},
        "swiss_roll": {"data_url1": "swiss_roll/domain1.txt", "data_url2": "swiss_roll/domain2.txt", 
                  "label_url1": "swiss_roll/discretelabel_domain1.txt", "label_url2": "swiss_roll/discretelabel_domain2.txt", 
                  "links": "swiss_roll/links.txt",
                  "scot_k": 10, "scot_e": 0.001, "smetric": "euclidean",
                  "sigma": 0.1},
        "circle": {"data_url1": "circle/domain1.txt", "data_url2": "circle/domain2.txt", 
                  "label_url1": "circle/classlabel_domain1.txt", "label_url2": "circle/classlabel_domain2.txt", 
                  "links": "circle/links.txt",
                  "scot_k": 10, "scot_e": 0.001, "smetric": "euclidean",
                  "sigma": 0.1},
        "benz": {"data_url1": "benz/domain1.txt", "data_url2": "benz/domain2.txt", 
                  "label_url1": "benz/label_domain1.txt", "label_url2": "benz/label_domain2.txt", 
                  "scot_k": 10, "scot_e": 0.001, "smetric": "euclidean",
                  "sigma": 0.1},
        "cross": {"data_url1": "cross/domain1.txt", "data_url2": "cross/domain2.txt", 
                  "label_url1": "cross/label_domain1.txt", "label_url2": "cross/label_domain2.txt", 
                  "scot_k": 10, "scot_e": 0.001, "smetric": "euclidean",
                  "sigma": 0.1},
    }
    return data_params


if __name__ == "__main__":
    np.seterr(under='ignore')

    apx = "/Users/zhouhan/Han/Research/Compbio/Project1/SONATA/data"

    dataset = datasets()
    for data, datapram in dataset.items():
        data_url1 = os.path.join(apx, datapram["data_url1"])
        data_url2 = os.path.join(apx, datapram["data_url2"])
        label_url1 = os.path.join(apx, datapram["label_url1"])
        label_url2 = os.path.join(apx, datapram["label_url2"])

        data1 = sonata.util.load_data(data_url1)
        data2 = sonata.util.load_data(data_url2)
        type1 = sonata.util.load_data(label_url1)
        type2 = sonata.util.load_data(label_url2)
        type1 = type1.astype(np.int32)
        type2 = type2.astype(np.int32)

        if "links" in datapram.keys():
            links = sonata.util.load_data(os.path.join(apx, datapram["links"]))
            links = links.astype(int)
        else:
            links = np.array(list(zip([i for i in range(data1.shape[0])], [i for i in range(data2.shape[0])])))
        print("data1 size={}\n data2 size={}".format(data1.shape, data2.shape))
        print("type1 size={}\n type2 size={}".format(type1.shape, type2.shape))

        basename_unioncom = f"../result/{data}/unioncom"
        basename_scot = f"../result/{data}/scot"

        ## norm 
        if "norm" in datapram.keys():
            data1 = sonata.util.wrapped_normalize(data1, norm=datapram["norm"])
            data2 = sonata.util.wrapped_normalize(data2, norm=datapram["norm"])
        ## denoise
        if data == "SNARE-seq":
            data1 = sonata.util.wrapped_pca(data1, n_components=10)

        ### evaluate SCOT by label transfer accuracy & FOSCTTM (run on CPU)
        ## 1. run multiple times
        n = int(sys.argv[1])
        test_multiround_scot(n=n, k=datapram["scot_k"], e=datapram["scot_e"], metric= datapram["smetric"], 
                             data1=data1, data2=data2, links=links, basename=basename_scot)



        # ### evaluate Unioncom by label transfer accuracy & FOSCTTM (run on GPU)
        # ## 1. run multiple times (setting different seeds)
        # test_multiround_unioncom(seed_range=[i for i in range(0, 500, 10)], data1=data1, data2=data2, links=links, basename=basename_unioncom)

        # ##2. run multiple parameters
        # test_multiparam_unioncom(data1, data2, links=links, basename=basename_unioncom, rho_range=[i for i in range(5, 16)])
        # test_multiparam_unioncom(data1, data2, links=links, basename=basename_unioncom, epsilon_range=[5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 1e-4])
        
