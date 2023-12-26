#!/usr/bin/env python
# coding: utf-8

import sonata

import numpy as np
np.seterr(under='ignore')
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')

if __name__=='__main__':
    data_url1 = "../data/scGEM/scGEM_expression.csv"
    data_url2 = "../data/scGEM/scGEM_methylation.csv"
    label_url1 = "../data/scGEM/scGEM_typeExpression.txt"
    label_url2 = "../data/scGEM/scGEM_typeMethylation.txt" 

    sigma=0.15

    ## load data
    data1 = sonata.util.load_data(data_url1) 
    data1 = sonata.util.wrapped_normalize(data1)

    ## sonata pipeline
    sn_instance = sonata.model.sonata(sigma=sigma)
    ambiguous_groups = sn_instance.check_ambiguity(data1)
    sonata_mapping_result = sn_instance.mapping_mat(data1, ambiguous_groups)

    # manifold aligner (SCOT pipeline)
    k = 25
    e = 0.005

    data2 = sonata.util.load_data(data_url2)
    data2 = sonata.util.wrapped_normalize(data2)
    scot_instance = sonata.scotv1.SCOT(data1, data2)
    scot_instance.align(k=k, e=e, metric="correlation", normalize=False)
    scot_instance.coupling

    # generate alternaltive mappings
    if ambiguous_groups:
        manifold_alternaltive_mappings = sn_instance.smap2amap(sonata_mapping_result, scot_instance.coupling)

    # visualization
    dm1_name = 'Gene Expression'
    dm2_name = 'DNA methylation'
    cell_labels = ["BJ", "d8", "d16T+", "d24T+", "iPS"]

    # visualize domain data
    label1 = sonata.util.load_data(label_url1)
    label2 = sonata.util.load_data(label_url2)
    sonata.vis.plt_domain_by_labels(data1, label1, color="#009ACD", title=dm1_name, y_tick_labels=cell_labels)
    sonata.vis.plt_domain_by_labels(data2, label2, color="#FF8C00", title=dm2_name, y_tick_labels=cell_labels)

    # visualize ambiguity groups
    sonata.vis.plt_ambiguous_groups(data1, ambiguous_groups)
    # visualize ambiguity cells
    sonata.vis.plt_cannotlink_by_labels(data1, label1, ambiguous_groups, sn_instance.ambiguous_links, y_tick_labels=cell_labels) 

    # visualize SCOT mapping
    x_aligned, y_aligned = sonata.util.projection_barycentric(scot_instance.X, scot_instance.y, scot_instance.coupling)
    sonata.vis.plt_mapping_by_labels(x_aligned, y_aligned, label1, label2, title1=dm1_name, title2=dm2_name, y_tick_labels=cell_labels)

    # visualize alternaltive mappings for manifold aligner
    if ambiguous_groups:
        sonata.vis.plt_alternaltive_mappings_by_label(scot_instance.X,  scot_instance.y, label1, label2, manifold_alternaltive_mappings, 
        title1=dm1_name, title2=dm2_name, y_tick_labels=cell_labels)
