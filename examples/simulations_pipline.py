#!/usr/bin/env python
# coding: utf-8

import sonata

import numpy as np
np.seterr(under='ignore')
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')

def datasets(dataset):
    data_params = {
        "t_branch": {"data_url1": "../data/t_branch/domain1.txt", "data_url2": "../data/t_branch/domain2.txt", 
                  "label_url1": "../data/t_branch/label_domain1.txt", "label_url2": "../data/t_branch/label_domain2.txt", 
                  "scot_k": 10, "scot_e": 0.001, "smetric": "euclidean",
                  "sigma": 0.1, "curve_style": True},
        "swiss_roll": {"data_url1": "../data/swiss_roll/domain1.txt", "data_url2": "../data/swiss_roll/domain2.txt", 
                  "label_url1": "../data/swiss_roll/label_domain1.txt", "label_url2": "../data/swiss_roll/label_domain2.txt", 
                  "scot_k": 10, "scot_e": 0.001, "smetric": "euclidean",
                  "sigma": 0.1},
        "circle": {"data_url1": "../data/circle/domain1.txt", "data_url2": "../data/circle/domain2.txt", 
                  "label_url1": "../data/circle/label_domain1.txt", "label_url2": "../data/circle/label_domain2.txt", 
                  "scot_k": 10, "scot_e": 0.001, "smetric": "euclidean",
                  "sigma": 0.1},
        "benz": {"data_url1": "../data/benz/domain1.txt", "data_url2": "../data/benz/domain2.txt", 
                  "label_url1": "../data/benz/label_domain1.txt", "label_url2": "../data/benz/label_domain2.txt", 
                  "scot_k": 10, "scot_e": 0.001, "smetric": "euclidean",
                  "sigma": 0.1},
        "cross": {"data_url1": "../data/cross/domain1.txt", "data_url2": "../data/cross/domain2.txt", 
                  "label_url1": "../data/cross/label_domain1.txt", "label_url2": "../data/cross/label_domain2.txt", 
                  "scot_k": 10, "scot_e": 0.001, "smetric": "euclidean",
                  "sigma": 0.1},
        # Add more modes and corresponding filenames as needed
    }
    return data_params[dataset]


if __name__=='__main__':
    for data in ['t_branch', 'swiss_roll', 'benz', 'cross', 'circle']:
        dataset = datasets(data)
        processor = sonata.pipline.SONATA_processor(dataset["data_url1"], dataset["data_url2"], dataset["label_url1"], dataset["label_url2"])
        processor.load_data()

        processor.vis_plt_domain_by_labels()

        processor.scot_mapping(k=dataset["scot_k"], e=dataset["scot_e"], metric=dataset["smetric"])

        processor.sonata_mapping(sigma=dataset["sigma"])