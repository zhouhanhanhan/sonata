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

        ## load data
        data1 = sonata.util.load_data(dataset["data_url1"])

        ## sonata pipline
        sn_instance = sonata.model.sonata(sigma=dataset["sigma"])
        ambiguous_groups = sn_instance.check_ambiguity(data1)
        sonata_mapping_result = sn_instance.mapping_mat(data1, ambiguous_groups)

        # manifold aligner (SCOT pipline)
        data2 = sonata.util.load_data(dataset["data_url2"])
        scot_instance = sonata.scotv1.SCOT(data1, data2)
        scot_instance.align(k=dataset["scot_k"], e=dataset["scot_e"], metric=dataset["smetric"], normalize=False)
        scot_instance.coupling

        # generate alternaltive mappings
        if ambiguous_groups:
            manifold_alternaltive_mappings = sn_instance.smap2amap(sonata_mapping_result, scot_instance.coupling)

        # visualization
        # visualize domain data
        label1 = sonata.util.load_data(dataset["label_url1"])
        label2 = sonata.util.load_data(dataset["label_url2"])
        sonata.vis.plt_domain_by_labels(data1, label1)
        sonata.vis.plt_domain_by_labels(data2, label2)

        # visualize ambiguity groups
        sonata.vis.plt_ambiguous_groups(data1, ambiguous_groups)
        # visualize ambiguity cells
        sonata.vis.plt_cannotlink_by_labels(data1, label1, ambiguous_groups, sn_instance.ambiguous_links) 

        # visualize SCOT mapping
        x_aligned, y_aligned = sonata.util.projection_barycentric(scot_instance.X, scot_instance.y, scot_instance.coupling)
        sonata.vis.plt_mapping_by_labels(x_aligned, y_aligned, label1, label2)

        # visualize alternaltive mappings for manifold aligner
        if ambiguous_groups:
            sonata.vis.plt_alternaltive_mappings_by_label(scot_instance.X,  scot_instance.y, label1, label2, manifold_alternaltive_mappings)
