#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.insert(1, '../src/')
import sonata

import numpy as np
np.seterr(under='ignore')
import warnings
warnings.filterwarnings("ignore")
import matplotlib
# matplotlib.use('Agg')
import sklearn

data_url1 = "../data/circle/domain1.txt"
data_url2 = "../data/circle/domain2.txt"
label_url1 = "../data/circle/label_domain1.txt"
label_url2 = "../data/circle/label_domain2.txt"

k = 10
e = 0.005
smode="connectivity"
smetric="euclidean" # correlation

sigma = 0.1

# In[2]:


data1 = sonata.util.load_data(data_url1)
data2 = sonata.util.load_data(data_url2)
label1 = sonata.util.load_data(label_url1)
label2 = sonata.util.load_data(label_url2)
print("data1 shape={}\tdata2 shape={}".format(data1.shape, data2.shape))
print("label1 shape={}\tlabel2 shape={}".format(label1.shape, label2.shape))

# ### 1. visualize two modalities

# In[3]:

# sonata.vis.plt_domain_by_labels(data1, label1, color="#009ACD", title=dm1_name, y_tick_labels=cell_labels)
# sonata.vis.plt_domain_by_labels(data2, label2, color="#FF8C00", title=dm2_name, y_tick_labels=cell_labels)

# sonata.vis.plt_domain_by_labels(data1, label1)
# sonata.vis.plt_domain_by_labels(data2, label2)

# sonata.vis.plt_domain(data1)
# sonata.vis.plt_domain(data2)

# ### 2. Mapping by SCOT (or any other manifold aligners)

# In[4]:
# k = 25
scot = sonata.scotv1.SCOT(data1.copy(), data2.copy())
scot.align(k = k, e=e, mode=smode, metric=smetric, normalize=False)
mapping = scot.coupling
x_aligned, y_aligned = sonata.util.projection_barycentric(scot.X, scot.y, mapping, XontoY = True)


# In[5]:

# sonata.vis.plt_mapping_by_labels(x_aligned, y_aligned, label1, label2, title1=dm1_name, title2=dm2_name, y_tick_labels=cell_labels)
# sonata.vis.plt_mapping_by_labels(x_aligned, y_aligned, label1, label2)

# ### 3. cell-cell alternaltive mappings from SONATA

# In[6]:


sn = sonata.sonata.sonata(kmin=10, sigma=sigma, t=0.1)
alter_mappings = sn.alt_mapping(data=data1) 


# #### 3.1 cell-cell ambiguities

# In[7]:


# sonata.vis.plt_cannotlink_by_labels(data1, label1, sn.ambiguous_nodes, sn.ambiguous_links, y_tick_labels=cell_labels)
sonata.vis.plt_cannotlink_by_labels(data1, label1, sn.ambiguous_nodes, sn.ambiguous_links, curve_style = False)

# #### 3.2 Ambiguous groups

# In[8]:


sonata.vis.plt_ambiguous_groups_by_labels(data1, sn.ambiguous_nodes, sn.cluster_labels)


# #### 3.3 Alternative alignments

# In[9]:


for idx, m in enumerate(alter_mappings, start=1):
    this_mapping = np.matmul(m, mapping)
    x_aligned, y_aligned = sonata.util.projection_barycentric(scot.X, scot.y, this_mapping, XontoY = True)
    # sonata.vis.plt_mapping_by_labels(x_aligned, y_aligned, label1, label2, title1=dm1_name, title2=dm2_name, y_tick_labels=cell_labels, XontoY=True)
    sonata.vis.plt_mapping_by_labels(x_aligned, y_aligned, label1, label2)

