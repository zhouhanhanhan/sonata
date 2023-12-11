# sonata
An `Internal Testing Use` version of `SONATA` for converting source code to BioMANIA APP.
The demo SONATA APP is [here](https://github.com/batmen-lab/BioMANIA/blob/main/examples/sonata_SNARE_seq.html).

## Requirements
Dependencies for **SONATA** are recorded in *requirements.txt*.  

For illustration, we downloaded Manifold aligner **SCOT**, a commonly-used manifold alignment method using optimal transport, from its [official github](https://github.com/rsinghlab/SCOT). In principle, SONATA is generalizable to any off-the-shelf manifold alignment methods.

## Data
All datasets are available at [this link](https://drive.google.com/drive/folders/1DKDP2eSfWODHiFqmn2GQY4m-sNda5seg?usp=sharing).

## Set up
```bash
git clone https://github.com/zhouhanhanhan/sonata.git
cd SONATA
pip install -e .
```

## Examples
Jupyter notebooks to replicate the results from the manuscript are available under folder *examples*:  
- Simulation datasets
    - no ambiguous: [simulation_swiss_roll.ipynb](https://github.com/batmen-lab/SONATA/blob/main/examples/simulation_swiss_roll.ipynb)
    - all ambiguous: [simulation_circle.ipynb](https://github.com/batmen-lab/SONATA/blob/main/examples/simulation_circle.ipynb)
    - partial ambiguous: [simulation_t_branch.ipynb](https://github.com/batmen-lab/SONATA/blob/main/examples/simulation_t_branch.ipynb), [simulation_benz.ipynb](https://github.com/batmen-lab/SONATA/blob/main/examples/simulation_benz.ipynb), [simulation_cross.ipynb](https://github.com/batmen-lab/SONATA/blob/main/examples/simulation_cross.ipynb)
- Real bio datasets: [scGEM.ipynb](https://github.com/batmen-lab/SONATA/blob/main/examples/scGEM.ipynb), [SNARE-seq.ipynb](https://github.com/batmen-lab/SONATA/blob/main/examples/SNARE-seq.ipynb)

Examples using pipline are available under folder *examples*:
- simulations_pipline.py
- scGEM_pipline.py
- SNARE-seq_pipline    

Example for basic use under folder *examples*: 
- simulations_basic.py


## How to use
### A quick start pipline
Input for SONATA pipline:  
 - *data_url1* and *data_url2*:  
 Recommended .txt or .csv file formats. Data file of the first and second domains that need to be aligned. The data should be in the form of numpy arrays or matrices, where each row represents a sample and each column represents a feature.
 - *label_url1* and *label_url2*:   
 Label data file of the first and second domains respectively.
 - *dm1_name* and *dm2_name*:   
 Dataset name for the first and second domains respectively. Optional to define for visualization.

```python
import sonata 

## initialization, optinal to define dm1_name and dm2_name for visualization!
processor = sonata.pipline.SONATA_processor(data_url1, data_url2, label_url1, label_url2)

## load data
processor.load_data()

## visualize domain data
processor.vis_plt_domain_by_labels()

## run manifold aligner SCOT
processor.scot_mapping(k, e, metric=smetric)

## run sonata to detect ambiguity and generate alternative solutions.
processor.sonata_mapping(sigma)
```

### Basic Use
Use sonata in a more flexible way.

Input for SONATA: *data* in form of numpy arrays/matrices, where the rows correspond to samples and columns correspond to features.
```python
import sonata
sn = sonata.sonata.sonata(kmin=10, sigma=0.1, t=0.1)
alter_mappings = sn.alter_mapping(data)
```

Run a manifold aligner, you can replace SCOT by your manifold aligner.
```python
scot = sonata.scotv1.SCOT(data1.copy(), data2.copy())
scot.align(k = k, e=e, mode=smode, metric=smetric, normalize=False)
mapping = scot.coupling
x_aligned, y_aligned = sonata.util.projection_barycentric(scot.X, scot.y, mapping, XontoY = True)
```

Generate and visualize alternaltive solutions.
```python
## visualize alternaltive solutions
for idx, m in enumerate(alter_mappings, start=1):
    this_mapping = np.matmul(m, mapping)
    x_aligned, y_aligned = sonata.util.projection_barycentric(scot.X, scot.y, this_mapping, XontoY = True)
    sonata.vis.plt_mapping_by_labels(x_aligned, y_aligned, label1, label2)

## visualize cell-cell ambiguities
sonata.vis.plt_cannotlink_by_labels(data1, label1, sn.ambiguous_nodes, sn.ambiguous_links, curve_style = False)

## visualize mutually ambiguity groups
sonata.vis.plt_ambiguous_groups_by_labels(data1, sn.ambiguous_nodes, sn.cluster_labels)

```

### Required parameters for sonata
- **k**: Number of neighbors to be used when constructing kNN graphs. Default=10. The number of neighbors k should be suffciently large to connect the corresponding k-NN graph   
- **sigma**: Bandwidth parameter for cell-wise ambiguity (Aij). Default=0.1.
- **t**: A threshold to ascertain the ambiguity status of individual cells before clustering them into groups. Default=0.1, with lower values resulting in stricter ambiguity classification.

### Optional parameters:
- **kmode**: Determine whether to use a connectivity graph (adjacency matrix of 1s/0s based on whether nodes are connected) or a distance graph (adjacency matrix entries weighted by distances between nodes). Default="distance"
- **kmetric**: Sets the metric to use while constructing nearest neighbor graphs. some possible choices are "euclidean", "correlation". Default= "euclidean".
- **kmax**: Maximum value of knn when constructing geodesic distance matrix. Default=200.
- **percnt_thres**: The percentile of the data distribution used in the calculation of the “virtual” cell. Default=95.
- **eval_knn**: Evaluate whether the alternative alignment distorts the data manifold by changing the mutual nearest neighbors of cells. Default=False.