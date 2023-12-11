#!/usr/bin/env python
# coding: utf-8

import sonata

import numpy as np
np.seterr(under='ignore')
import warnings
warnings.filterwarnings("ignore")
import matplotlib
# matplotlib.use('Agg')

data_url1 = "../data/scGEM/scGEM_expression.csv"
data_url2 = "../data/scGEM/scGEM_methylation.csv"
label_url1 = "../data/scGEM/scGEM_typeExpression.txt"
label_url2 = "../data/scGEM/scGEM_typeMethylation.txt"    

dm1_name = 'Gene Expression'
dm2_name = 'DNA methylation'
cell_labels = ["BJ", "d8", "d16T+", "d24T+", "iPS"]

processor = sonata.pipline.SONATA_processor(data_url1, data_url2, label_url1, label_url2, dm1_name, dm2_name)

processor.load_data()
processor.normalize_data()

processor.vis_plt_domain_by_labels(stick_label_name=cell_labels)


k = 25
e = 0.005
processor.scot_mapping(k, e, stick_label_name=cell_labels)

sigma=0.15
processor.sonata_mapping(sigma, stick_label_name=cell_labels)
