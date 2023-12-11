from . import util
from . import vis
from . import scotv1
from . import sonata
import os

import numpy as np
np.seterr(under='ignore')
import warnings
warnings.filterwarnings("ignore")

class SONATA_processor:
    """
    A pipline for SONATA.

    Parameters
    ----------
    data_path1 : str
        Data path of domain 1.
    data_path2 : str
        Data path of domain 2.
    label_path1 : str, optional
        Label path of domain 1.
    label_path2 : str, optional
        Label path of domain 2.
    dm1_name : str, optional
        Dataset name of domain 1, for visualization.
    dm2_name : str, optional
        Dataset name of domain 2, for visualization.

    Basic Use:
    ----------   
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

    Returns
    -------
    None
    """
    def __init__(self, data_path1, data_path2, label_path1 = None, label_path2 = None, dm1_name = None, dm2_name = None) -> None:
        self.initialize_class(data_path1, data_path2, label_path1, label_path2, dm1_name, dm2_name)

    def initialize_class(self, data_path1:str, data_path2:str, label_path1:str=None, label_path2:str=None, dm1_name:str=None, dm2_name:str=None) -> None:
        """
        Initialize input data paths.

        Parameters
        ----------
        data_path1 : str
            Data path of domain 1.
        data_path2 : str
            Data path of domain 2.
        label_path1 : str, optional
            Label path of domain 1.
        label_path2 : str, optional
            Label path of domain 2.
        dm1_name : str, optional
            Dataset name of domain 1, for visualization.
        dm2_name : str, optional
            Dataset name of domain 2, for visualization.

        Returns
        -------
        None
        """
        self.data_path1 = data_path1
        self.data_path2 = data_path2
        self.label_path1 = label_path1
        self.label_path2 = label_path2
        self.dm1_name = dm1_name if dm1_name else "Domain 1"
        self.dm2_name = dm2_name if dm2_name else "Domain 2"

    def load_data(self) -> None:
        """
        Load data from specified data paths and optionally load labels.

        Returns
        -------
        None        
        """
        self.data1 = util.load_data(self.data_path1)
        self.data2 = util.load_data(self.data_path2)
        print("data1 shape={}\tdata2 shape={}".format(self.data1.shape, self.data2.shape))
        if self.label_path1 and self.label_path2:
            self.label1 = util.load_data(self.label_path1)
            self.label2 = util.load_data(self.label_path2)
            print("label1 shape={}\tlabel2 shape={}".format(self.label1.shape, self.label2.shape))

    def normalize_data(self, norm:str='l2', norm_data:str='') -> None:
        """
        Normalize the data using the specified normalization method.

        Parameters
        ----------
        norm : str, optional
            The normalization method to use, by default 'l2'.
        norm_data : str, optional
            Which dataset to normalize, by default '', normalize all datasets.

        Returns
        -------
        None
        """
        if norm_data=="data1":
            self.data1 = util.wrapped_normalize(self.data1, norm=norm)
        elif norm_data=="data2":
            self.data2 = util.wrapped_normalize(self.data2, norm=norm)
        else:
            self.data1 = util.wrapped_normalize(self.data1, norm=norm)
            self.data2 = util.wrapped_normalize(self.data2, norm=norm)

    def apply_pca(self, n_components:int, proj_data:str) -> None:
        """
        Denising data by applying pca.

        Parameters
        ----------
        n_components : str
            The number of components to keep during PCA.
        proj_data : str
            The data on which to apply PCA denoising. Specify as 'data1', 'data2', 
            or any other value to apply to all datasets.         

        Returns
        -------
        None
        """
        self.s_data1 = self.data1.copy()
        self.s_data2 = self.data2.copy()
        # Apply PCA
        self.s_data1 = self.s_data1.astype('float64')
        self.s_data2 = self.s_data2.astype('float64')
        pca_instance = util.Wrapped_PCA(n_components=n_components)
        if proj_data=="data1":
            pca_instance.fit(self.s_data1)
            self.s_data1 = pca_instance.fit_transform(self.s_data1)
        elif proj_data=="data2":
            pca_instance.fit(self.s_data2)
            self.s_data2 = pca_instance.fit_transform(self.s_data2)
        else:
            pca_instance.fit(self.s_data1)
            self.s_data1 = pca_instance.fit_transform(self.s_data1)
            pca_instance.fit(self.data2)
            self.s_data2 = pca_instance.fit_transform(self.s_data2)

    def vis_plt_domain_by_labels(self, color1:str="#009ACD", color2:str="#FF8C00", a1:float=0.8, a2:float=0.8, stick_label_name:str=None) -> None:
        """
        Visualize PCA scatter plot of both domain based on the labels.

        Parameters
        ----------
        color1 : str, optional
            The color used for plotting the domain of data1, by default "#009ACD".
        color2 : str, optional
            The color used for plotting the domain of data1, by default "#FF8C00". 
        a1 : float, optional
            The transparency rate for scatters in data1, by default 0.8.
        a2 : float, optional
            The transparency rate for scatters in data2, by default 0.8.
        stick_label_name : str, optional
            The label names to display on the y-axis. If None, label names will be set as in the label file, by default None.

        Returns
        -------
        None
        """
        vis.plt_domain_by_labels(self.data1, self.label1, color=color1, title=self.dm1_name, y_tick_labels=stick_label_name, a=a1)
        vis.plt_domain_by_labels(self.data2, self.label2, color=color2, title=self.dm2_name, y_tick_labels=stick_label_name, a=a2)

    def scot_mapping(
            self, 
            k:int, 
            e:float, 
            mode:str="connectivity", 
            metric:str="correlation", 
            norm:str=None,
            norm_data:str='', 
            n_pca:int=None,
            proj_data:str='',
            XontoY:bool=False, 
            stick_label_name:str=None,
            a:float=0.8,
            c1: str = "#FF8C00",
            c2: str = "#009ACD",
            gc1: str = "Blues",
            gc2: str = "Oranges") -> None:
        """
        Apply SCOT mapping to align and visualize the data based on labels.

        Parameters
        ----------
        k : int
            The number of nearest neighbors used in the SCOT algorithm.
        e : float
            Regularization constant for entropic regularization.
        mode : str, optional
            Type of graph ('connectivity' or 'distance') in the SCOT algorithm, by default "connectivity".
        metric : str, optional
            Metric for constructing nearest neighbor graphs in the SCOT algorithm, by default "correlation".
        norm : str, optional
            The normalization method to use, possible values are 'l2', 'l1' 'max' and None. If None, no normalization will be applied, by default None.
        norm_data : str, optional
            Which dataset to normalize, by default '', normalize all datasets.
        n_pca : str, optional
            The number of components to keep during PCA.
        proj_data : str
            The data on which to apply PCA denoising. Specify as 'data1', 'data2', 
            or any other value to apply to all datasets.           
        XontoY : bool, optional
            Specify whether to align data1 onto data2, by default False.
        stick_label_name : str or None, optional
            The label names to display on the y-axis. If None, label names will be set as in the label file, by default None.
        a : float, optional
            The transparency rate for scatters, by default 0.8.
        c1 : str, optional
            The color for the first domain in subplot1, by default '#FF8C00'.
        c2 : str, optional
            The color for the second domain in subplot1, by default '#009ACD'. 
        gc1 : str, optional
            The gradient label color for the first domain in subplot2 when too many label types, by default 'Blues'.
        gc2 : str, optional
            The gradient label color for the second domain in subplot2 when too many label types, by default 'Oranges'.  
        
        Returns
        -------
        None

        Notes
        -----
        This function performs SCOT mapping to align self.data1 and self.data2 using the SCOT algorithm. The alignment is
        performed based on the specified parameters k, e, mode, metric, and norm. The aligned data points are then projected
        onto a lower dimensional space using the barycentric projection. The resulting projections are visualized using
        labels stored in self.label1 and self.label2.
        """
        if norm:
            self.normalize_data(norm=norm, norm_data = norm_data)
        if n_pca:
            self.apply_pca(n_components=n_pca, proj_data=proj_data)
            scot_data1 = self.s_data1
            scot_data2 = self.s_data2
        else:
            scot_data1 = self.data1
            scot_data2 = self.data2

        self.scot_instance = scotv1.SCOT(scot_data1, scot_data2)
        self.scot_instance.align(k=k, e=e, mode=mode, metric=metric, normalize=False)
        x_aligned, y_aligned = util.projection_barycentric(self.scot_instance.X, self.scot_instance.y, self.scot_instance.coupling, XontoY = XontoY)
        vis.plt_mapping_by_labels(
            x_aligned, y_aligned, self.label1, self.label2, title1=self.dm1_name, title2=self.dm2_name,
            y_tick_labels=stick_label_name, XontoY=XontoY, 
            a=a, c1=c1, c2=c2, gc1=gc1, gc2=gc2
            )

    def sonata_mapping(
            self, 
            sigma:float, 
            kmin:int=10, 
            t:float=0.1,
            kmax:int=200, 
            kmode:str="distance", 
            kmetric:str="euclidean", 
            percnt_thres:int=95, 
            eval_knn:bool=True,
            sonata_dm:str="data1",
            XontoY:bool=False, 
            cl_alpha: float = 0.1, 
            curve_style : bool = False,
            stick_label_name:str=None,
            a: float = 0.8,
            c1: str = "#FF8C00",
            c2: str = "#009ACD",
            gc1: str = "Blues",
            gc2: str = "Oranges",
            max_return: int=4,
            vs_link: bool=True,
            vs_group: bool=True,
            norm:str=None,
            norm_data:str='', 
            n_pca:int=None,
            proj_data:str='') -> None:
        """
        Apply SONATA mapping to visualize the data based on ambiguous nodes and links.
        By default SONATA will detect self-ambiguity in the first data, set sonata_dm="data2" if you want to detect in another data.

        Parameters
        ---------
        Required parameters for SONATA:
        sigma : float
            Bandwidth parameter for cell-wise ambiguity (Aij) in the SONATA algorithm.
        kmin : int, optional
             The minimum number of neighbors to connect the k-NN graph in the SONATA algorithm, by default 10.
        t : float, optional
            Threshold to ascertain the ambiguity status of individual cells before clustering them into groups in the SONATA algorithm, by default 0.1.
        
        Optional parameters for SONATA:
        kmax : int, optional
            The maximum number of neighbors to connect the k-NN graph in the SONATA algorithm, by default 200.
        kmode : str, optional
            Mode to use for calculating the k-NN graph in the SONATA algorithm, either 'connectivity' or 'distance', 
            adjacency matrix of a connectivity graph is based on whether nodes are connected and 
            a distance graph is based on wighted distances between nodes, by default "distance".
        kmetric : str, optional
            Metric to use for calculating the k-NN graph, possible choices are 'euclidean' and 'correlation', by default "euclidean".
        percnt_thres : int, optional
            Percentile of the data distribution used in the calculation of the “virtual” cell in the SONATA algorithm, by default 95.
        eval_knn : bool, optional
            Specify whether to evaluate the alternative alignment distorts the data manifold in the SONATA algorithm, by default False.
        sonata_dm : str, optional
            The data on which SONATA detects self-ambiguity, by default "data1".
        
        Visualization parameters:
        XontoY : bool, optional
            Specify whether to align data1 onto data2 in the barycentric projection, by default False.
        cl_alpha : float, optional
            The transparency rate for plotting the cannot-link connections, by default 0.1.
        curve_style : bool, optional
            Specify whether to use curved style for plotting the cannot-link connections, by default False.
        stick_label_name : str or None, optional
            The label names to display on the y-axis, by default None.
        a : float, optional
            The transparency rate for plotting the ambiguous groups, by default 0.8.
        c1 : str, optional
            The color for plotting data1, by default '#FF8C00'.
        c2 : str, optional
            The color for plotting data2, by default '#009ACD'.
        gc1 : str, optional
            The colormap for data1 in the barycentric projection, by default 'Blues'.
        gc2 : str, optional
            The colormap for data2 in the barycentric projection, by default 'Oranges'.
        max_return : int, optional
            Set the maximum number of alternative solution plots to return. A value of -1 will return all solutions, by default 4.
        vs_cell : bool, optional
            Specify whether to visualize cell-cell ambiguity links, by default True.
        vs_group : bool, optional
            Specify whether to visualize mutually ambiguity groups, by default True.

        Data preprocess parameters:
        norm : str, optional
            The normalization method to use, possible values are 'l2', 'l1' 'max' and None. If None, no normalization will be applied, by default None.
        norm_data : str, optional
            Which dataset to normalize, by default '', normalize all datasets.
        n_pca : str, optional
            The number of components to keep during PCA.
        proj_data : str
            The data on which to apply PCA denoising. Specify as 'data1', 'data2', 
            or any other value to apply to all datasets. 

        Returns
        -------
        None

        Notes
        -----
        - The SONATA mapping is performed using the sonata.sonata method with the specified parameters sigma, kmin, t, kmax,
        kmode, kmetric, percnt_thres, and eval_knn.
        - The ambiguous nodes and links are obtained from self.sn_instance.alt_mapping using the self.data1.
        - The resulting ambiguous groups and mappings are visualized using the vis.plt_cannotlink_by_labels and 
        vis.plt_ambiguous_groups_by_labels methods.
        - For each mapping in self.sonata_mapping_result, the barycentric projection is applied to align the self.scot_instance.X 
        and self.scot_instance.y based on the mapping. The resulting aligned projections are visualized using the 
        vis.plt_mapping_by_labels method.
        """
        if norm:
            self.normalize_data(norm=norm, norm_data = norm_data)
        if n_pca:
            self.apply_pca(n_components=n_pca, proj_data=proj_data)
            sonata_data1 = self.s_data1
            sonata_data2 = self.s_data2
        else:
            sonata_data1 = self.data1
            sonata_data2 = self.data2

        self.sn_instance = sonata.sonata(kmin=kmin, sigma=sigma, t=t, kmax=kmax, kmode=kmode, kmetric=kmetric, percnt_thres=percnt_thres, eval_knn=eval_knn)
        self.sonata_mapping_result = self.sn_instance.alt_mapping(data=sonata_data1 if sonata_dm=="data1" else sonata_data2)

        if self.sonata_mapping_result == None: 
            """
            Unambiguous situation, no need to plot ambiguity related figs.
            """
            return
        
        if vs_link:
            vis.plt_cannotlink_by_labels(
                    self.data1, self.label1, self.sn_instance.ambiguous_nodes, self.sn_instance.ambiguous_links, 
                    y_tick_labels=stick_label_name, cl_alpha=cl_alpha, curve_style=curve_style
                ) 
        if vs_group:
            vis.plt_ambiguous_groups_by_labels(
                    self.data1, self.sn_instance.ambiguous_nodes, 
                    self.sn_instance.cluster_labels, alpha=a
                )
        
        for idx, m in enumerate(self.sonata_mapping_result, start=1):
            if max_return != -1 and idx > max_return:
                continue
            this_mapping = np.matmul(m, self.scot_instance.coupling)
            x_aligned, y_aligned = util.projection_barycentric(self.scot_instance.X, self.scot_instance.y, this_mapping, XontoY = False)
            vis.plt_mapping_by_labels(
                x_aligned, y_aligned, self.label1, self.label2, title1=self.dm1_name, title2=self.dm2_name, 
                y_tick_labels=stick_label_name, XontoY=XontoY, a=a, c1=c1, c2=c2, gc1=gc1, gc2=gc2)

        print("Sonata finds {} alternative solutions. Plotting {} of them. If you want to visualize all alternaltive solutions, set max_return=-1."
              .format(idx, idx if idx < max_return else max_return))
        
if __name__=='__main__':
    data_url1 = "../data/scGEM/scGEM_expression.csv"
    data_url2 = "../data/scGEM/scGEM_methylation.csv"
    label_url1 = "../data/scGEM/scGEM_typeExpression.txt"
    label_url2 = "../data/scGEM/scGEM_typeMethylation.txt"    

    dm1_name = 'Gene Expression'
    dm2_name = 'DNA methylation'
    cell_labels = ["BJ", "d8", "d16T+", "d24T+", "iPS"]

    processor = SONATA_processor(data_url1, data_url2, label_url1, label_url2, dm1_name, dm2_name, cell_labels)

    processor.load_data()
    processor.normalize_data()

    processor.vis_plt_domain_by_labels()

    n_components = 10
    proj_data = "data2"
    processor.apply_pca(n_components, proj_data)

    k = 25
    e = 0.005
    processor.scot_mapping(k, e)

    sigma=0.15
    processor.sonata_mapping(sigma)