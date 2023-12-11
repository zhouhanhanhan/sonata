#!/usr/bin/env python
# coding: utf-8

import sonata

import numpy as np
np.seterr(under='ignore')
import warnings
warnings.filterwarnings("ignore")
import matplotlib
# matplotlib.use('Agg')



data_url1 = "../data/SNARE-seq/scatac_feat.npy"
data_url2 = "../data/SNARE-seq/scrna_feat.npy"
label_url1 = "../data/SNARE-seq/SNAREseq_atac_types.txt"
label_url2 = "../data/SNARE-seq/SNAREseq_rna_types.txt"    

dm1_name = 'Chromatin Accessibility'
dm2_name = 'Gene Expression'
cell_labels = ["H1", "GM", "BJ", "K562"]

processor = sonata.pipline.SONATA_processor(data_url1, data_url2, label_url1, label_url2, dm1_name, dm2_name)

processor.load_data()
processor.normalize_data()

processor.vis_plt_domain_by_labels(stick_label_name=cell_labels)

k = 110
e = 0.001
scot_pca = 10
processor.scot_mapping(k, e, stick_label_name=cell_labels, n_pca=scot_pca)


sigma=0.15
sonata_pca = 2
processor.sonata_mapping(sigma, stick_label_name=cell_labels, n_pca=sonata_pca, vs_link=False)
