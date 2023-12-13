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
- SNARE-seq_pipline.py

Example for basic use under folder *examples*: 
- simulations_basic.py


## How to use
### Basic use
```python
import sonata
# load data
data = sonata.util.load_data(data_url)
# data preprocess (if necessary)
data = sonata.util.wrapped_normalize(data, norm='l2')
```
#### SONATA
- Input: *data* numpy arrays, where each row represents a sample and each column represents a feature.  
- Output: *ambiguous_groups* dictionary containing the groups of ambiguous cells identified in the input data. 

```python
sn_instance = sonata.model.sonata(sigma=0.15)
ambiguous_groups = sn_instance.check_ambiguity(data)
```
#### Required parameters for SONATA
- **sigma**: Bandwidth parameter for cell-wise ambiguity (Aij). Default=0.1.
- **kmin**: Number of neighbors to be used when constructing kNN graphs. Default=10. The number of neighbors k should be suffciently large to connect the corresponding k-NN graph.   
- **t**: A threshold to ascertain the ambiguity status of individual cells before clustering them into groups. Default=0.1, with lower values resulting in stricter ambiguity classification.

#### Optional parameters:
- **kmode**: Determine whether to use a connectivity graph (adjacency matrix of 1s/0s based on whether cells are connected) or a distance graph (adjacency matrix entries weighted by distances between cells). Default="distance"
- **kmetric**: Sets the metric to use while constructing nearest neighbor graphs. some possible choices are "euclidean", "correlation". Default= "euclidean".
- **kmax**: Maximum value of knn when constructing geodesic distance matrix. Default=200.
- **percnt_thres**: The percentile of the data distribution used in the calculation of the “virtual” cell. Default=95.
- **eval_knn**: Evaluate whether the alternative alignment distorts the data manifold by changing the mutual nearest neighbors of cells. Default=True.

### Using SONATA to generate self-ambiguity mappings
- Input: 
  - *data*: cell by feature numpy array matrix.
  - *ambiguous_groups*: identified by SONATA
- Output: 
  - *sonata_mappings*: generator of self-alternaltive cell by cell mapping matrices. 
```python
sonata_mappings = sn_instance.mapping_mat(data, ambiguous_groups)
```

### Using SONATA to generate alternaltive solutions for manifold aligners
Run a manifold aligner
```python
# load data
data1 = sonata.util.load_data(data_url1)
data2 = sonata.util.load_data(data_url2)

# manifold aligner (SCOT as an example)
scot_instance = sonata.scotv1.SCOT(data1, data2)
scot_instance.align(k=10, e=1e-3)

# the cell by cell mapping matrix of SCOT:
scot_instance.coupling
```
Generate alternaltive solutions
```python
# generate alternaltive mappings
manifold_alternaltive_mappings = sn_instance.smap2amap(sonata_mappings, scot_instance.coupling)
```
