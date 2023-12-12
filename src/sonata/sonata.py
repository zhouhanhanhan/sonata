import numpy as np

from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp 
from scipy.spatial.distance import cdist

from scipy.interpolate import UnivariateSpline
from scipy.interpolate import splrep, PPoly

from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans

import itertools
import ot
import typing

class sonata(object):
    """
    SONATA algorithm for disambiguating manifold alignment of single-cell data
    https://www.biorxiv.org/content/10.1101/2023.10.05.561049v2

    Input for SONATA: data in form of numpy arrays/matrices, where the rows correspond to samples and columns correspond to features.
    Basic Use:
    import sonata
    sn = sonata.sonata.sonata(kmin=10, sigma=0.1, t=0.1)
    alter_mappings = sn.alter_mapping(data)

    Required parameters
    - k: Number of neighbors to be used when constructing kNN graphs. Default=10. The number of neighbors k should be suffciently large to connect the corresponding k-NN graph   
    - sigma: Bandwidth parameter for cell-wise ambiguity (Aij). Default=0.1.
    - t: A threshold to ascertain the ambiguity status of individual cells before clustering them into groups. Default=0.1, with lower values resulting in stricter ambiguity classification.

    Optional parameters:
    - kmode: Determine whether to use a connectivity graph (adjacency matrix of 1s/0s based on whether nodes are connected) or a distance graph (adjacency matrix entries weighted by distances between nodes). Default="distance"
    - kmetric: Sets the metric to use while constructing nearest neighbor graphs. some possible choices are "euclidean", "correlation". Default= "euclidean".
    - kmax: Maximum value of knn when constructing geodesic distance matrix. Default=200.
    - percnt_thres: The percentile of the data distribution used in the calculation of the “virtual” cell. Default=95.
    - eval_knn: Evaluate whether the alternative alignment distorts the data manifold by changing the mutual nearest neighbors of cells. Default=False.    
    """

    def __init__(self, kmin=10, sigma=0.1, t=0.1, kmax=200, kmode="distance", kmetric="euclidean", percnt_thres=95, eval_knn=False) -> None:
        self.initialize_class(kmin, sigma, t, kmax, kmode, kmetric, percnt_thres, eval_knn)
    def initialize_class(self, kmin:int=10, sigma:float=0.1, t:float=0.1, kmax:int=200, kmode:str="distance", kmetric:str="euclidean", percnt_thres:int=95, eval_knn:bool=False) -> None:
        """
        Initialize sonata instance with given parameters.

        Parameters
        ----------
        kmin : int, optional
            The minimum number of neighbors to connect in the k-NN graph, by default 10.
        sigma : float, optional
            Bandwidth parameter for cell-wise ambiguity (Aij), by default 0.1.
        t : float, optional
            A threshold to ascertain the ambiguity status of individual cells before clustering them into groups, by default 0.1.
        kmax : int, optional
            The maximum number of neighbors to connect in the k-NN graph, by default 200.
        kmode : str, optional
            Mode to use for calculating the k-NN graph, either 'connectivity' or 'distance', 
            adjacency matrix of a connectivity graph is based on whether nodes are connected and a distance graph is based on wighted distances between nodes, by default "distance".
        kmetric : str, optional
            Metric to use for calculating the k-NN graph, possible choices are 'euclidean' and 'correlation', by default "euclidean".
        percnt_thres : int, optional
            The percentile of the data distribution used in the calculation of the “virtual” cell, by default 95.
        eval_knn : bool, optional
            Evaluate whether the alternative alignment distorts the data manifold by changing the mutual nearest neighbors of cells, by default False.

        Returns
        -------
        None
        """
        self.kmin = kmin
        self.kmax = kmax
        self.kmode = kmode
        self.kmetric = kmetric

        self.sigma = sigma
        self.percnt_thres = percnt_thres
        self.t = t
        self.eval_knn = eval_knn

        self.geo_mat = None
        self.knn = None
        self.l1_mat = None
        self.cell_amat = None
        self.group_amats = None

        # for plt
        self.ambiguous_links = None
        self.ambiguous_nodes = None
        self.cluster_labels = None

        # for elbow methods
        self.K = None
        self.K_yerror = None
        self.K_xstep = None

    def alt_mapping(self, data:np.ndarray) -> typing.Generator[np.ndarray, None, None]:
        """
        Pipline to perform alternative mappings.

        Parameters
        ----------
        data : np.ndarray
            Input data for which the alternaltive mappings should be performed.

        Returns
        -------
        group_amats : Generator
            The generator of the alternaltive mapping matrices.
        """
        # cell-wise ambiguity
        self.construct_graph(data)
        self.l1_mat = self.geo_similarity(geo_dist=self.geo_mat)
        self.cell_ambiguity()

        # group-wise ambiguity
        self.group_amats = self.group_ambiguity(data)

        return self.group_amats
    
    
    def construct_graph(self, data:np.ndarray) -> None:
        """
        Constructing k-NN graph and calculating geodesic distance.

        Parameters
        ----------
        data : np.ndarray
            Input data for which the k-NN graph should be constructed.

        Returns
        -------
        None

        Notes
        -----
        kmax/kmin: k should be sufficiently large to connect the corresponding k-NN graph.
        kmode: 'connectivity'/'distance'
        metric: 'euclidean'/'correlation'
        """
        print('constructing knn graph ...')
        nbrs = NearestNeighbors(n_neighbors=self.kmin, metric=self.kmetric, n_jobs=-1).fit(data)
        knn = nbrs.kneighbors_graph(data, mode = self.kmode)

        connected_components = sp.csgraph.connected_components(knn, directed=False)[0]
        while connected_components != 1:
            if self.kmin > np.max((self.kmax, 0.01*len(data))):
                break
            self.kmin += 2
            nbrs = NearestNeighbors(n_neighbors=self.kmin, metric=self.kmetric, n_jobs=-1).fit(data)
            knn = nbrs.kneighbors_graph(data, mode = self.kmode)
            connected_components = sp.csgraph.connected_components(knn, directed=False)[0]
        print('final k ={}'.format(self.kmin))

        # calculate the shortest distance between nodes
        dist = sp.csgraph.floyd_warshall(knn, directed=False)
        
        dist_max = np.nanmax(dist[dist != np.inf])
        dist[dist > dist_max] = 2*dist_max
        
        # global max-min normalization
        norm_geo_dist = (dist - np.min(dist)) / (np.max(dist) - np.min(dist))

        self.geo_mat = norm_geo_dist
        self.knn = knn
    
    def geo_similarity(self, geo_dist: np.ndarray) -> np.ndarray:
        """
        Calculate the L1 distance similarity matrix for geodesic distances.

        Parameters
        ----------
        geo_dist : numpy.ndarray
            An array representing geodesic distances.

        Returns
        -------
        numpy.ndarray
            The L1 distance similarity matrix.

        Notes
        -----
        This function calculates the L1 distance similarity matrix for the given geodesic distances.
        It sorts the distances and then computes the L1 distance similarity matrix using the 'cityblock' metric.
        The resulting matrix represents the geodesic similarity between different cells.

        """
        sorted_geo_dist = np.sort(geo_dist, axis=1)
        l1_dist = cdist(sorted_geo_dist, sorted_geo_dist, 'cityblock') / sorted_geo_dist.shape[1]
        return l1_dist
    
    def cell_similarity(self, mat: np.ndarray) -> np.ndarray:
        """
        Calculate initial cell-wise ambiguity matrix.

        Parameters
        ----------
        mat : numpy.ndarray
            Input matrix for L1 distance similarity calculation.

        Returns
        -------
        numpy.ndarray
            The cell-wise ambiguity matrix.

        Notes
        -----
        This function computes a initial cell-wise ambiguity matrix from the l1 similarity matrix.
        It utilizes a soft-max function with a bandwidth parameter, 'self.sigma', to facilitate a small portion of cells being ambiguous. 
        Additionally, it applies 0-1 normalization to ensure the cell-wise ambiguity in 0-1 range.
        The resulting matrix represents the ambiguity between different cells, 
        with value ~0 indicating low ambiguity and ~1 indicating higher ambiguity.

        """
        d_matrix = mat/np.power(self.sigma, 2)
        d_matrix_e = np.exp(-d_matrix)
        d_matrix_sum = np.sum(d_matrix_e, axis = 1).reshape(d_matrix_e.shape[0],1)
        cell_amat = d_matrix_e/d_matrix_sum
        # normalize cell-wise ambiguity to 0-1 range
        cell_amat = (cell_amat - np.min(cell_amat, axis=1, keepdims=True)) / (np.max(cell_amat, axis=1, keepdims=True) -
                                                                    np.min(cell_amat, axis=1, keepdims=True))
        return cell_amat

    def cell_ambiguity(self):
        """
        Calculate cell-wise ambiguity.

        Notes
        -----
        This function calculates cell-wise ambiguity using a series of steps:
        1. It computes an initial cell-wise similarity matrix 'init_cell_amat' using the 'self.cell_similarity' function.
        2. It then applies a safeguard to the ambiguity calculation using the 'self.safe_ambiguity' function.
        3. Finally, it calibrates the ambiguity using the 'self.fit_spline' function and stores the result in 'self.cell_amat'.

        """
        print('calculating cell-wise ambiguity ...')
        # ambiguity
        init_cell_amat = self.cell_similarity(self.l1_mat)

        # ambiguity safeguard
        cell_amat_safe = self.safe_ambiguity(self.l1_mat)

        # ambiguity calibration
        cell_amat_clean = self.fit_spline(cell_amat_safe)

        self.cell_amat = cell_amat_clean

    def safe_ambiguity(self, mat: np.ndarray) -> np.ndarray:
        """
        Apply a safeguard to cell-wise ambiguity calculation.

        Parameters
        ----------
        mat : numpy.ndarray
            Input matrix for ambiguity calculation.

        Returns
        -------
        numpy.ndarray
            The safeguarded cell-wise ambiguity matrix.

        Notes
        -----
        This function applies a safeguard to the ambiguity calculation process.
        It involves shuffling the input matrix, computing a shuffled L1 distance similarity matrix,
        and comparing percentiles to control ambiguity.
        The safeguarded cell-wise ambiguity matrix is then returned.

        """
        n = self.geo_mat.shape[0]
        geo_mat_shuffled = self.geo_mat.copy().flatten()
        np.random.shuffle(geo_mat_shuffled)
        geo_mat_shuffled = geo_mat_shuffled.reshape(self.geo_mat.shape)
        l1_mat_shuffled = self.geo_similarity(geo_mat_shuffled)

        percent_l1 = np.percentile(mat, q=self.percnt_thres, axis=1)
        percent_l1_shuffled = np.percentile(l1_mat_shuffled, q=self.percnt_thres, axis=1)
        control_vec = percent_l1 * np.sign(percent_l1 - percent_l1_shuffled)
        l1_mat_aug = np.zeros((n+1, n+1))
        l1_mat_aug[:n, :n] = mat
        l1_mat_aug[n, :n] = control_vec
        l1_mat_aug[:n, n] = control_vec
        cell_amat_safe = self.cell_similarity(l1_mat_aug)

        cell_vec_aug = cell_amat_safe[:n, n]
        cell_amat_safe = cell_amat_safe[:n, :n]
        # Ambiguous value should not be higher than baseline (cell_vec_aug)
        for node_idx in range(n):
            cell_amat_safe[node_idx, :] = np.maximum(cell_amat_safe[node_idx, :], cell_vec_aug[node_idx])
        return cell_amat_safe

    def fit_spline(self, cell_amat: np.ndarray, r: int = 10) -> np.ndarray:
        """
        Fit a spline to mitigate the distance-related biases of cell-wise ambiguity.

        Parameters
        ----------
        cell_amat : numpy.ndarray
            The cell-wise ambiguity matrix.
        r : int, optional
            The smoothing radius for averaging nearest neighbors, by default 10.

        Returns
        -------
        numpy.ndarray
            The cell-wise ambiguity matrix without distance-related biases.

        Notes
        -----
        This function fits a spline to the cell-wise ambiguity matrix to mitigate the distance-related biases.
        It fits the univariate spline using scipy package after smoothing values within a specified radius 'r'. 
        Then the neighborhood noises are removed.
        The resulting matrix represents the ambiguity values without distance-related biases.

        """
        cell_amat_clean = np.copy(cell_amat)
        for node_id in range(cell_amat.shape[0]):
            tuple_arr = list(zip(self.geo_mat[node_id, :], cell_amat[node_id, :], ))
            tuple_arr.sort(key=lambda x: x[0])

            geo_arr = np.asarray([x[0] for x in tuple_arr], dtype=float)
            Y_arr = np.asarray([x[1] for x in tuple_arr], dtype=float)
            X_arr = np.asarray(list(range(len(tuple_arr))), dtype=float)
 
            # smoothen cell-wise ambiguity by averaging nearest neighbors with radius = r
            Y_arr_smooth = np.copy(Y_arr)
            for mid_idx in range(len(Y_arr)):
                Y_arr_smooth[mid_idx] = np.mean(Y_arr[max(0, mid_idx-r):min(len(Y_arr), mid_idx+r)])

            # remove noise for all-ambiguous datasets
            Y_arr_smooth = np.where(Y_arr_smooth < 0.1, np.round(Y_arr_smooth, 1), Y_arr_smooth)
            Y_arr = Y_arr_smooth

            # fitting spline 
            spline = UnivariateSpline(X_arr, Y_arr, k = 4) 

            # find minimal inflection point of the curve
            dv1 = spline.derivative(n = 1) # 1st derivative
            y_dv1 = dv1(X_arr)
            tck = splrep(X_arr, y_dv1)
            ppoly = PPoly.from_spline(tck)
            dv1_roots = ppoly.roots(extrapolate=False) # 1st derivative = 0
            dv2 = spline.derivative(n = 2) # 2nd derivative

            # remove ambiguous neighbors by detecting 1st minimal inflection point
            curve_m = np.where(dv2(dv1_roots) > 0)[0]

            if len(curve_m) > 0:
                idx = int(dv1_roots[curve_m[0]])
                cell_amat_clean[node_id, self.geo_mat[node_id, :] <= geo_arr[idx]] = np.min(cell_amat_clean[node_id, :])
        
        return cell_amat_clean


    def group_ambiguity(self, data: np.ndarray)  -> typing.Generator[np.ndarray, None, None]:
        """
        Calculate group-wise ambiguity and provide the alternaltive mappings.

        Parameters
        ----------
        data : numpy.ndarray
            The input data.

        Returns
        -------
        map_mat : Generator
            The generator of the alternaltive mapping matrices.
            
        Notes
        -----
        This function calculates group-wise ambiguity, which involves selecting ambiguous nodes,
        finding ambiguous groups and generates matrices of alternative mappings.
        The resulting generator generates matrices of alternaltive mappings.

        """
        print('calculating group-wise ambiguity ...')
        ambiguous_nodes = self.select_ambiguous_nodes()
        unambiguous_nodes = np.setdiff1d(list(range(data.shape[0])), ambiguous_nodes)
        self.ambiguous_nodes = ambiguous_nodes

        if len(ambiguous_nodes) == 0:
            print("There is no ambiguity")
            map_mat = None
        else:
            ambiguous_node_groups = self.find_ambiguous_groups(data, ambiguous_nodes)
            map_mat = self.map_ambiguous_groups(data, ambiguous_node_groups, unambiguous_nodes)

        return map_mat
        

    def select_ambiguous_nodes(self) -> np.ndarray:
        """
        Select ambiguous nodes.

        Returns
        -------
        numpy.ndarray
            An array of indices representing ambiguous nodes.

        Notes
        -----
        This function selects ambiguous nodes based on a threshold 'self.t'
        applied to the cell-wise ambiguity matrix.

        """
        cell_amat_copy = self.cell_amat.copy()
        n = cell_amat_copy.shape[0]
        cell_amat_copy[cell_amat_copy <= self.t] = 0
        cell_amat_sum_arr = cell_amat_copy.sum(axis=1)
        ambiguous_nodes = np.where(cell_amat_sum_arr > 0)[0]
        return ambiguous_nodes

    def find_ambiguous_groups(self, data: np.ndarray, ambiguous_nodes: np.ndarray) -> list:
        """
        Find ambiguous groups using semi-supervised clustering.

        Parameters
        ----------
        data : numpy.ndarray
            The input data.
        ambiguous_nodes : numpy.ndarray
            An array of indices representing ambiguous nodes.

        Returns
        -------
        list
            A list of lists, where each inner list contains indices of nodes in an ambiguous group.

        Notes
        -----
        This function uses semi-supervised clustering with cannot-link constraints to find ambiguous groups.
        It returns a list of ambiguous node groups based on the clustering results.

        using semi-supervised-clustering with cannot link
        code: https://github.com/datamole-ai/active-semi-supervised-clustering

        """
        
        data_mat_ambiguous = data[ambiguous_nodes, :]
        cell_amat_ambiguous = self.cell_amat[ambiguous_nodes, :][:, ambiguous_nodes]

        # ambiguity threshold
        ambiguous_indices = np.where(cell_amat_ambiguous > self.t)
        
        # all cell-wise ambiguity
        cannot_links = list(zip(ambiguous_indices[0], ambiguous_indices[1]))
        
        # group ambiguous nodes  -- elbow method to choose k
        print('deciding best k for clustering ...')
        if self.K == None:
            self.elbow_k(data_mat_ambiguous, cannot_links, k_range=10)
            print('K = {} groups choosen by elbow method'.format(self.K))
        clusterer = PCKMeans(n_clusters=self.K)
        clusterer.fit(data_mat_ambiguous, cl=cannot_links)
        labels = np.asarray(clusterer.labels_, dtype=int)

        ambiguous_node_groups = []
        for class_label in np.unique(labels):
            class_indices = np.where(labels == class_label)[0]
            ambiguous_node_groups.append(ambiguous_nodes[class_indices])
        
        self.ambiguous_links = cannot_links
        self.cluster_labels = labels
        return ambiguous_node_groups
 
    def elbow_k(self, data: np.ndarray, cannot_link: list, k_range: int = 10):
        """
        Determine the optimal number of clusters (k) using the elbow method.

        Parameters
        ----------
        data : numpy.ndarray
            The input data.
        cannot_link : list
            A list of tuples representing cannot-link constraints.
        k_range : int, optional
            The range of k values to consider, by default 10.

        Notes
        -----
        This function determines the optimal number of clusters (k) using the elbow method.
        It calculates the number of ambiguity pairs for each k and selects the k with the sharpest decrease in ambiguity pairs.

        """
        from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans
        cl_arr = np.transpose(np.array(cannot_link))

        # calculate # of ambiguity pairs for each k
        y_error = []
        for k in range(1, k_range):
            clusterer = PCKMeans(n_clusters=k)
            clusterer.fit(data, cl=cannot_link)

            labels = clusterer.labels_
            n_pairs = 0
            for label in np.unique(labels):
                this_cluster = np.where(labels == label)[0]
                n_pairs += np.sum(np.logical_and(np.isin(cl_arr[0], this_cluster), 
                                                            np.isin(cl_arr[1], this_cluster)))
            y_error.append(n_pairs)

        # normalize error
        y_error = np.array(y_error)/np.square(data.shape[0])
        # normalize x axis
        x_step = 1/data.shape[0]

        # choose the best k by 2nd derivative
        best_k = np.argmax(self.second_grad(y_error, step = x_step)) + 2

        self.K = best_k
        self.K_yerror = y_error
        self.K_xstep = x_step

    def map_ambiguous_groups(self, data: np.ndarray, ambiguous_node_groups: list, unambiguous_nodes: np.ndarray) -> typing.Generator[np.ndarray, None, None]:
        """
        Generate alternative mappings.

        Parameters
        ----------
        data : numpy.ndarray
            The input data.
        ambiguous_node_groups : list
            A list of lists, where each inner list contains indices of nodes in an ambiguous group.
        unambiguous_nodes : numpy.ndarray
            An array of indices representing unambiguous nodes.

        Returns
        -------
        map_mat : Generator
            The generator of the alternaltive mapping matrices.

        Notes
        -----
        This function generates alternaltive mappings by aligning ambiguous cells using optimal transport.
        It returns a generator generating the matrices of alternaltive mappings.

        """
        np.seterr(under='ignore')
        ambiguous_nodes = np.asarray([], dtype=int)
        for group in ambiguous_node_groups:
            ambiguous_nodes = np.concatenate((ambiguous_nodes, group))
        assert data.shape[0] == len(ambiguous_nodes)+len(unambiguous_nodes)

        ### evaluate valid group
        if self.eval_knn:
            valid_perm = self.eval_valid_group_knn(ambiguous_node_groups, unambiguous_nodes)
        else:
            valid_perm= list(itertools.permutations(range(len(ambiguous_node_groups))))[1:]
        print("all vaild perms are: ", valid_perm)

        assert len(ambiguous_node_groups) > 1
        for perms in valid_perm:
            # keep the diagonal for original nodes
            map_mat = np.zeros_like(self.geo_mat)
            print("perms: ", perms)

            # keep the diagonal for unambiguous nodes
            for node in unambiguous_nodes: map_mat[node, node] = 1.0        

            for i in range(len(ambiguous_node_groups)):
                group_idx1 = i
                group_idx2 = perms[i]

                node_group1 = ambiguous_node_groups[group_idx1]
                node_group2 = ambiguous_node_groups[group_idx2]

                if group_idx1 == group_idx2:
                    # keep the diagonal for unchanged nodes
                    for node in np.concatenate((node_group1, node_group2)): map_mat[node, node] = 1.0 
                else:
                    print('changed group id: ', group_idx1, group_idx2)
                    # aligning ambiguous group pairs
                    cost = 1-self.cell_amat
                    cost = cost[node_group1, :][:, node_group2]
                    p1 = ot.unif(len(node_group1))
                    p2 = ot.unif(len(node_group2))
                    T = ot.emd(p1, p2, cost)
                    T = normalize(T, norm='l1', axis=1, copy=False)

                    # fill in the T matrix for mapped nodes 
                    for group1_idx in range(len(node_group1)):
                        group1_node = node_group1[group1_idx]
                        for group2_idx in range(len(node_group2)):
                            group2_node = node_group2[group2_idx]
                            map_mat[group1_node, group2_node] = T[group1_idx, group2_idx]

            yield map_mat


    # second graduate for k elbow
    def second_grad(self, k_arr: list, step: float) -> np.ndarray:
        """
        Calculate the second gradient for the elbow method.

        Parameters
        ----------
        k_arr : list
            A list of ambiguity pairs for different values of k.
        step : float
            The step size for normalization.

        Returns
        -------
        second_grad : numpy.ndarray
            The second gradient values.

        Notes
        -----
        This function calculates the second gradient for the elbow method
        to help determine the optimal number of clusters (k).

        """
        # first grad
        first_grad = (k_arr[1:] - k_arr[:-1])/step 
        # 2nd grad
        second_grad = (first_grad[1:] - first_grad[:-1])/(1+first_grad[1:]*first_grad[:-1])
        second_grad = np.arctan(np.abs(second_grad))

        return second_grad

    def eval_valid_group_knn(self, ambiguous_node_groups: list, unambiguous_nodes: np.ndarray) -> list:
        """
        Evaluate valid groups based on k-nearest neighbors.

        Parameters
        ----------
        ambiguous_node_groups : list
            A list of lists, where each inner list contains indices of nodes in an ambiguous group.
        unambiguous_nodes : numpy.ndarray
            An array of indices representing unambiguous nodes.

        Returns
        -------
        valid_perm : list
            A list of valid permutations for ambiguous groups.

        Notes
        -----
        This function evaluates valid groups based on k-nearest neighbors
        and returns a list of valid permutations for ambiguous groups.

        """
        knn = self.knn.tocoo()
        knn_arr = knn.toarray()

        connected_groups = []
        for tup in list(itertools.combinations(range(len(ambiguous_node_groups)), 2)):
            group_idx1, group_idx2 = tup[0], tup[1]
            node_group1 = ambiguous_node_groups[group_idx1]
            node_group2 = ambiguous_node_groups[group_idx2]
            sub_knn_arr = knn_arr[node_group1, :][:, node_group2]
            if np.sum(sub_knn_arr) > 0.0:
                connected_groups.append(tup)
        print("connected_groups_ambiguous: ", connected_groups)

        for group_idx1 in range(len(ambiguous_node_groups)):
            group_idx2 = -1 # unambiguous nodes
            node_group1 = ambiguous_node_groups[group_idx1]
            sub_knn_arr = knn_arr[node_group1, :][:, unambiguous_nodes]
            if np.sum(sub_knn_arr) > 0:
                connected_groups.append((group_idx1, group_idx2))
        connected_groups = np.array(connected_groups)
        print('connected_groups_all: ', connected_groups)   
        
        valid_perm = []
        sorted_connected_groups = np.sort(connected_groups, axis = 1)
        for perms in list(itertools.permutations(range(len(ambiguous_node_groups))))[1:]:
            new_connected_groups = np.copy(connected_groups)
            for i in range(len(ambiguous_node_groups)):
                group_idx1 = i
                group_idx2 = perms[i]
                if group_idx1 != group_idx2:
                    new_connected_groups[connected_groups == group_idx1] = group_idx2  
            sorted_new_connected_groups = np.sort(new_connected_groups, axis = 1)  

            if sorted(sorted_connected_groups.tolist()) == sorted(sorted_new_connected_groups.tolist()):
                valid_perm.append(perms)

        return valid_perm
