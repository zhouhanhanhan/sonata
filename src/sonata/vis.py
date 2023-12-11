import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plt_domain(data_mat: np.ndarray, color: str = '#009ACD', title: str = None, save_url: str = None, marker: str = '.', alpha: float = 0.8, show: bool = True) -> None:
    """
    Scatter plot of data points in two-dimensional space.

    Parameters
    ----------
    data_mat : np.ndarray
        The data matrix containing data points.
    color : str
        The color of the markers.
    title : str
        The title of the plot.
    marker : str, optional
        The marker style, by default '.'.
    alpha : float, optional
        The transparency of the markers, by default 0.8.
    show : bool, optional
        Whether to display the plot or save the plot, by default True.
    save_url : str, optional
        The path to save the plot as an image file, by default None.

    Returns
    -------
    None
    """
    pca = PCA(n_components=2).fit(data_mat)
    data_embed = pca.fit_transform(data_mat)

    fig = plt.figure(figsize=(4,4))
    plt.scatter(data_embed[:,0], data_embed[:,1], c = color, marker = marker, s=25, alpha= alpha)

    plt.xlabel("Principal component 1", fontsize=10)
    plt.ylabel("Principal component 2", fontsize=10) 
    plt.tick_params(axis='x', labelsize=8)
    plt.tick_params(axis='y', labelsize=8)
    if title:
        plt.title(title, fontdict={'fontsize': 15})
    if show:
        plt.show()
    else:
        assert (save_url is not None), "Please specify save_url!"
        os.makedirs(os.path.dirname(save_url), exist_ok=True)
        plt.savefig(save_url)
    plt.close()

def plt_domain_by_labels(data_mat: np.ndarray, label: np.ndarray, color: str = '#009ACD', title: str = None, y_tick_labels: list = None, save_url: str = '', a: float = 0.8, show: bool = True) -> None:
    """
    Scatter plot of data points colored by labels in two-dimensional space. 
    Subplot1 : unlabeled data scatter   subplot2 : data scatter colored by labels.

    Parameters
    ----------
    data_mat : np.ndarray
        The data matrix containing data points.
    label : np.ndarray
        The labels associated with each data point.
    color : str, optional
        The color values for the markers in subplot1.
    title : str, optional
        The title of the plot.
    y_tick_labels : list, optional
        The labels for the colorbar ticks in subplot2.
    save_url : str, optional
        The path to save the plot as an image file, by default ''.
    a : float, optional
        The alpha value for marker transparency in subplot2, by default 0.8.
    show : bool, optional
        Whether to display the plot, by default True.

    Returns
    -------
    None
    """
    pca = PCA(n_components=2).fit(data_mat)
    data_embed = pca.fit_transform(data_mat)

    uniq_label = np.unique(label)
    colormap = plt.get_cmap('rainbow', len(uniq_label))
    norm = mpl.colors.BoundaryNorm(np.arange(0.5,len(uniq_label)+1.5), colormap.N) 
    
    # Subplot1: domain scatters
    fig = plt.figure(figsize=(9, 4))
    ax0 = plt.subplot(1,2,1)
    plt.scatter(data_embed[:,0], data_embed[:,1], c = color, s=25, alpha = a) 
    plt.xlabel("Principal component 1", fontsize=10)
    plt.ylabel("Principal component 2", fontsize=10)
    plt.tick_params(axis='x', labelsize=8)
    plt.tick_params(axis='y', labelsize=8)
    plt.title('Colored by domain', fontdict={'size': 10})

    # Subplot2: domain scatters colored by label
    ax1 = plt.subplot(1,2,2)
    if len(uniq_label) > 10:
        print("Too many labels, use gradient color instead.")
        scatter = plt.scatter(data_embed[:,0], data_embed[:,1], c = label, s=25, alpha= a, cmap = plt.cm.get_cmap('Blues'))
    else:
        scatter = plt.scatter(data_embed[:,0], data_embed[:,1], c = label, s=25, cmap=colormap, norm=norm, alpha = a) 
    x_min = np.min(data_embed[:, 0])
    x_max = np.max(data_embed[:, 0])
    plt.xlim(x_min-0.03*(x_max-x_min), x_max+0.1*(x_max-x_min))   
    plt.xlabel("Principal component 1", fontsize=10)
    plt.ylabel("Principal component 2", fontsize=10)
    plt.tick_params(axis='x', labelsize=8)
    plt.tick_params(axis='y', labelsize=8)
    plt.title('Colored by label', fontdict={'size': 10})
    cbaxes = inset_axes(ax1, width="3%", height="100%", loc='right') 
    if y_tick_labels:
        cbar = plt.colorbar(cax=cbaxes, orientation='vertical')
        cbar.set_ticks(ticks=(np.arange(0,len(uniq_label))+1), labels=y_tick_labels)
    elif len(uniq_label) > 10:
        cbar = plt.colorbar(scatter, cax=cbaxes, orientation='vertical')
        cbar.set_ticks([])
        cbar.ax.set_yticklabels([])
    else:
        cbar = plt.colorbar(scatter, cax=cbaxes, orientation='vertical')
        cbar.set_ticks(ticks=(np.arange(0,len(uniq_label))+1), labels=uniq_label) 
          

    if title:
        plt.suptitle(title, fontdict={'size': 15})
    plt.subplots_adjust(wspace=0.5)
    if show:
        plt.show()
    else:
        assert (save_url is not None), "Please specify save_url!"
        os.makedirs(os.path.dirname(save_url), exist_ok=True)
        plt.savefig(save_url, bbox_inches="tight")
    plt.close()


def plt_mapping_by_label(X_new: np.ndarray, y_new: np.ndarray, label1: np.ndarray, label2: np.ndarray, save_url: str = '', a1: float = 0.3, a2: float = 0.8, show: bool = False) -> None:
    """
    Scatter plot of data points from two domains aligned by SCOT, colored by labels.

    Parameters
    ----------
    X_new : np.ndarray
        Data points from the first domain.
    y_new : np.ndarray
        Data points from the second domain.
    label1 : np.ndarray
        Labels associated with data points from the first domain.
    label2 : np.ndarray
        Labels associated with data points from the second domain.
    save_url : str, optional
        The path to save the plot as an image file, by default ''.
    a1 : float, optional
        The alpha value for marker transparency in the first domain, by default 0.3.
    a2 : float, optional
        The alpha value for marker transparency in the second domain, by default 0.8.
    show : bool, optional
        Whether to display the plot, by default False.

    Returns
    -------
    None
    """
    data_mat = np.concatenate((X_new, y_new), axis=0)
    pca = PCA(n_components=2).fit(data_mat)
    data_embed = pca.fit_transform(data_mat)
    X_proj=data_embed[0: X_new.shape[0],]
    y_proj=data_embed[X_new.shape[0]:,]

    markers = ['.', '*', '+', '1', '|']
    all_label = np.unique(np.concatenate((label1, label2), axis = 0))

    fig = plt.figure(figsize=(3, 3))
    for i in range(len(all_label)):
        mark = markers[i]
        label = all_label[i]
        plt.scatter(y_proj[:,0][np.where(label2 ==label)], y_proj[:,1][np.where(label2 ==label)], 
                    c = "#FF8C00", marker = mark, s=150, alpha = a2, label="domain2")
    for i in range(len(all_label)):
        mark = markers[i]
        label = all_label[i]    
        plt.scatter(X_proj[:,0][np.where(label1 ==label)], X_proj[:,1][np.where(label1 ==label)], 
                    c = "#009ACD", marker = mark, s=100, alpha= a1, label="domain1")
        
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", prop={'size': 10})
    plt.xlabel("Principal component 1", fontsize=10)
    plt.ylabel("Principal component 2", fontsize=10)
    plt.tick_params(axis='x', labelsize=8)
    plt.tick_params(axis='y', labelsize=8)
    plt.title("Domains Aligned by SCOT", fontdict={'fontsize': 15})
    if show:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_url), exist_ok=True)
        plt.savefig(save_url, bbox_inches="tight")
    plt.close()

def plt_mapping_by_labels(
    X_new: np.ndarray,
    y_new: np.ndarray,
    label1: np.ndarray,
    label2: np.ndarray,
    title1: str = "Domain 1",
    title2: str = "Domain 2",
    y_tick_labels: list = None,
    a: float = 0.8,
    c1: str = "#FF8C00",
    c2: str = "#009ACD",
    gc1: str = "Blues",
    gc2: str = "Oranges",
    save_url: str = None,
    XontoY: bool = True,
    show: bool = True,
) -> None:
    """
    Scatter plot of data points from two domains aligned with labels.

    Parameters
    ----------
    X_new : np.ndarray
        Data points from the first domain.
    y_new : np.ndarray
        Data points from the second domain.
    label1 : np.ndarray
        Labels for data points in the first domain.
    label2 : np.ndarray
        Labels for data points in the second domain.
    title1 : str, optional
        Title for the first domain, by default 'Domain 1'.
    title2 : str, optional
        Title for the second domain, by default 'Domain 2'.
    y_tick_labels : list, optional
        Labels for color bar ticks, by default None.
    a : float, optional
        The alpha value for marker transparency in subplot2, by default 0.8.
    c1 : str, optional
        The color for the first domain in subplot1, by default '#FF8C00'.
    c2 : str, optional
        The color for the second domain in subplot1, by default '#009ACD'. 
    gc1 : str, optional
        The gradient label color for the first domain in subplot2 when too many label types, by default 'Blues'.
    gc2 : str, optional
        The gradient label color for the second domain in subplot2 when too many label types, by default 'Oranges'.   
    save_url : str, optional
        The path to save the plot as an image file, by default None.
    XontoY : bool, optional
        Whether to plot X onto Y (True) or Y onto X (False), by default True.
    show : bool, optional
        Whether to display the plot, by default True.

    Returns
    -------
    None
    """
    
    data_mat = np.concatenate((X_new, y_new), axis=0)
    pca = PCA(n_components=2).fit(data_mat)
    data_embed = pca.fit_transform(data_mat)

    X_proj=data_embed[0: X_new.shape[0],]
    y_proj=data_embed[X_new.shape[0]:,]

    all_label = np.unique(np.concatenate((label1, label2), axis = 0))
    colormap = plt.get_cmap('rainbow', len(all_label))
    norm = mpl.colors.BoundaryNorm(np.arange(0.5,len(all_label)+1.5), colormap.N) 

    #Plot aligned domains, samples colored by domain identity:
    fig = plt.figure(figsize=(9, 4))
    ax0 = plt.subplot(1,2,1)
    if XontoY:
        plt.scatter(y_proj[:,0], y_proj[:,1], c = c1, s = 25, label = title2, alpha = a)    
        plt.scatter(X_proj[:,0], X_proj[:,1], c = c2, s = 25, label = title1, alpha = a)
    else:
        plt.scatter(X_proj[:,0], X_proj[:,1], c = c2, s=25, label = title1, alpha = a)        
        plt.scatter(y_proj[:,0], y_proj[:,1], c = c1, s=25, label = title2, alpha = a)  
    plt.legend(prop={'size': 10})
    plt.xlabel("Principal component 1", fontsize=10)
    plt.ylabel("Principal component 2", fontsize=10)
    plt.tick_params(axis='x', labelsize=8)
    plt.tick_params(axis='y', labelsize=8)
    plt.title('Colored by domain', fontdict={'size': 10})
    

    ax1 = plt.subplot(1,2,2)
    colormap = plt.get_cmap('rainbow', len(all_label))   
    
    if len(all_label) > 10:
        print("Too many labels, use gradient color instead.")
        if XontoY:
            plt.scatter(y_proj[:,0], y_proj[:,1], c = label2, s=25, alpha = a, label = title2, cmap = plt.cm.get_cmap(gc2))
            plt.scatter(X_proj[:,0], X_proj[:,1], c = label1, s=25, alpha = a, label = title1, cmap = plt.cm.get_cmap(gc1))
        else:
            plt.scatter(X_proj[:,0], X_proj[:,1], c = label1, s=25, alpha = a, label = title1, cmap = plt.cm.get_cmap(gc1))
            plt.scatter(y_proj[:,0], y_proj[:,1], c = label2, s=25, alpha = a, label = title2, cmap = plt.cm.get_cmap(gc2))
    else:
        if XontoY:
            plt.scatter(y_proj[:,0], y_proj[:,1], c=label2, s=25, cmap=colormap, label = title2, alpha = a)    
            scatter = plt.scatter(X_proj[:,0], X_proj[:,1], c=label1, s=25, cmap=colormap, label = title1, alpha = a)
        else:            
            scatter = plt.scatter(X_proj[:,0], X_proj[:,1], c=label1, s=25, cmap=colormap, label = title1, alpha = a)        
            plt.scatter(y_proj[:,0], y_proj[:,1], c=label2, s=25, cmap=colormap, label = title2, alpha = a) 

    x_min = np.min(data_embed[:, 0])
    x_max = np.max(data_embed[:, 0])
    plt.xlim(x_min-0.03*(x_max-x_min), x_max+0.1*(x_max-x_min))      
    plt.xlabel("Principal component 1", fontsize=10)
    plt.ylabel("Principal component 2", fontsize=10)
    plt.tick_params(axis='x', labelsize=8)
    plt.tick_params(axis='y', labelsize=8)
    plt.title('Colored by label', fontdict={'size': 10})
    if y_tick_labels:
        cbaxes = inset_axes(ax1, width="3%", height="100%", loc='right') 
        cbar = plt.colorbar(cax=cbaxes, orientation='vertical')
        cbar.set_ticks(ticks=(np.arange(0,len(all_label))+1), labels=y_tick_labels)
    elif len(all_label) <= 10:
        cbaxes = inset_axes(ax1, width="3%", height="100%", loc='right') 
        cbar = plt.colorbar(scatter, cax=cbaxes, orientation='vertical')
        cbar.set_ticks(ticks=(np.arange(0,len(all_label))+1), labels=all_label)      
    

    plt.suptitle('Aligned Domains', fontdict={'size': 15}) 
    plt.subplots_adjust(wspace=0.5)
    if show:
        plt.show()
    else:
        assert (save_url is not None), "Please specify save_url!"
        os.makedirs(os.path.dirname(save_url), exist_ok=True)
        plt.savefig(save_url, bbox_inches="tight")
    plt.close()

def plt_cannotlink_by_labels(
    data_mat: np.ndarray, 
    labels: np.ndarray,
    ambiguous_nodes: np.ndarray, 
    ambiguous_links: list, 
    y_tick_labels: str = None, 
    save_url: str = None, 
    cl_alpha: float = 0.1, 
    curve_style : bool = False,
    show=True
) -> None:
    """
    Scatter plot of data points with all cell-cell ambiguities. 

    Parameters
    ----------
    data_mat : np.ndarray
        The data matrix containing data points.
    label : np.ndarray
        The labels associated with each data point.
    ambiguous_nodes : np.ndarray
        The array of ambiguous node ids.
    ambiguous_links : list
        The list of cell-cell ambuguities.
    y_tick_labels : list, optional
        The labels for the colorbar ticks.
    save_url : str, optional
        The path to save the plot as an image file, by default None.
    cl_alpha : float, optional
        The alpha value for ambiguous_links transparency, by default 0.1.
    curve_style : bool, optional
        Whether to use curve-style links instead of strainght lines, by default False.
    show : bool, optional
        Whether to display the plot, by default True.

    Returns
    -------
    None
    """    
    pca = PCA(n_components=2).fit(data_mat)
    data_embed = pca.fit_transform(data_mat)

    uniq_label = np.unique(labels)
    colormap = plt.get_cmap('rainbow', len(uniq_label))
    norm = mpl.colors.BoundaryNorm(np.arange(0.5,len(uniq_label)+1.5), colormap.N) 

    fig = plt.figure(figsize=(6, 6))
    ax0 = plt.subplot(1,1,1)
    if len(uniq_label) > 10:
        print("Too many labels, use gradient color instead.")
        scatter = plt.scatter(data_embed[:,0], data_embed[:,1], c = labels, label="Node", s=25, cmap=plt.cm.get_cmap('Blues'), zorder=10)
    else:
        scatter = plt.scatter(data_embed[:,0], data_embed[:,1], c = labels, label="Node", s=25, cmap=colormap, norm=norm, zorder=10)

    # plt ambiguous_links
    data_pca_ambiguous = data_embed[ambiguous_nodes, :]
    ambiguous_links = np.transpose(np.array(ambiguous_links))
    if curve_style:
        for i in range(len(ambiguous_links[0])):
            rad = 0.4 if data_pca_ambiguous[ambiguous_links[0][i], 1] > 0 else -0.4
            plt.annotate("",
                    xy=([data_pca_ambiguous[ambiguous_links[0][i], 0], data_pca_ambiguous[ambiguous_links[0][i], 1]]),
                    xytext=(data_pca_ambiguous[ambiguous_links[1][i], 0], data_pca_ambiguous[ambiguous_links[1][i], 1]),
                    size=20, va="center", ha="center",
                    arrowprops=dict(color='black',
                                    arrowstyle="-",
                                    connectionstyle="arc3, rad={}".format(rad),
                                    linewidth=0.2,
                                    alpha = cl_alpha
                                    )
                    )
    else:
        cl_alpha = cl_alpha/10 if ambiguous_links.shape[1]/ambiguous_nodes.shape[0] > 100 else cl_alpha
        plt.plot([data_pca_ambiguous[ambiguous_links[0], 0], data_pca_ambiguous[ambiguous_links[1], 0]], 
                [data_pca_ambiguous[ambiguous_links[0], 1], data_pca_ambiguous[ambiguous_links[1], 1]], 
                c="grey", alpha = cl_alpha, linewidth = 0.2)
    
    x_min = np.min(data_embed[:, 0])
    x_max = np.max(data_embed[:, 0])
    plt.xlim(x_min-0.03*(x_max-x_min), x_max+0.1*(x_max-x_min))   
    plt.xlabel("Principal component 1", fontsize=10)
    plt.ylabel("Principal component 2", fontsize=10)
    plt.tick_params(axis='x', labelsize=8)
    plt.tick_params(axis='y', labelsize=8)
    plt.title("cell-cell ambiguities", fontdict={'fontsize': 15})
    cbaxes = inset_axes(ax0, width="3%", height="100%", loc='right') 
    cbar = plt.colorbar(cax=cbaxes, orientation='vertical')
    cbar.set_ticks(ticks=(np.arange(0,len(uniq_label))+1), labels=y_tick_labels)   
    if y_tick_labels:
        cbar = plt.colorbar(cax=cbaxes, orientation='vertical')
        cbar.set_ticks(ticks=(np.arange(0,len(uniq_label))+1), labels=y_tick_labels)
    elif len(uniq_label) > 10:
        cbar = plt.colorbar(scatter, cax=cbaxes, orientation='vertical')
        cbar.set_ticks([])
        cbar.ax.set_yticklabels([]) 
    else:
        cbar = plt.colorbar(scatter, cax=cbaxes, orientation='vertical')
        cbar.set_ticks(ticks=(np.arange(0,len(uniq_label))+1), labels=uniq_label) 
 
    if show:
        plt.show() 
    else:
        assert (save_url is not None), "Please specify save_url!"
        os.makedirs(os.path.dirname(save_url), exist_ok=True)
        plt.savefig(save_url, bbox_inches="tight")
    plt.close()


def plt_ambiguous_groups_by_labels(
    data_mat: np.ndarray,
    ambiguous_nodes: np.ndarray,
    ambiguous_labels: np.ndarray,
    save_url: str = '',
    alpha: float = 0.8,
    show: bool = True,
) -> None:
    """
    Scatter plot of ambiguous groups with labels.

    Parameters
    ----------
    data_mat : np.ndarray
        Data points.
    ambiguous_nodes : np.ndarray
        Indices of ambiguous nodes.
    ambiguous_labels : np.ndarray
        Biological labels for ambiguous nodes.
    save_url : str, optional
        The path to save the plot as an image file, by default ''.
    alpha : float, optional
        Alpha value for data points, by default 0.8.
    show : bool, optional
        Whether to display the plot, by default True.

    Returns
    -------
    None
    """
    assert len(ambiguous_nodes) == len(ambiguous_labels)
    pca = PCA(n_components=2).fit(data_mat)
    data_embed = pca.fit_transform(data_mat)
    data_pca_ambiguous = data_embed[ambiguous_nodes, :]
    unambuguous_nodes = np.setdiff1d(list(range(data_mat.shape[0])), ambiguous_nodes)

    fig = plt.figure(figsize=(6, 6))
    plt.scatter(list(data_embed[unambuguous_nodes, 0]), list(data_embed[unambuguous_nodes, 1]), 
                c="grey", alpha=alpha, label='Certain', s=25)
    for idx, class_label in enumerate(np.unique(ambiguous_labels)):
        class_indices = np.where(ambiguous_labels == class_label)[0]
        plt.scatter(list(data_pca_ambiguous[class_indices, 0]), list(data_pca_ambiguous[class_indices, 1]), alpha=alpha,
                    label='Ambiguous Group={}'.format(class_label), s=25)
        
    plt.legend(prop={'size': 10})
    plt.xlabel("Principal component 1", fontsize=10)
    plt.ylabel("Principal component 2", fontsize=10)
    plt.tick_params(axis='x', labelsize=8)
    plt.tick_params(axis='y', labelsize=8)
    plt.title("Ambiguous groups", fontdict={'fontsize': 10})
    if show:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_url), exist_ok=True)
        plt.savefig(save_url, bbox_inches="tight")
    plt.close()

def plt_k_elbow(x_step: float, yerror: list, best_k: int, save_url: str = '', show: bool = False) -> None:
    """
    Create an Elbow Method plot for choosing the optimal number of clusters.

    Parameters
    ----------
    x_step : float
        Step size for the x-axis values (number of clusters).
    yerror : list
        List of error values for different numbers of clusters.
    best_k : int
        The optimal number of clusters determined by the Elbow Method.
    save_url : str, optional
        The file path where the plot should be saved, by default '' (no saving).
    show : bool, optional
        Whether to display the plot (True) or save it (False), by default False.

    Returns
    -------
    None
        This function does not return any value; it either displays or saves the plot.

    Notes
    -----
    The Elbow Method is a technique for selecting the optimal number of clusters for a clustering algorithm. 
    This function creates a plot of the error values for different numbers of clusters and highlights the 
    optimal number of clusters.

    Example
    -------
    >>> x_step = 1
    >>> yerror = [10, 5, 3, 2, 1]
    >>> best_k = 3
    >>> save_url = 'elbow_plot.png'
    >>> plt_k_elbow(x_step, yerror, best_k, save_url, show=True)
    """
    k_range = len(yerror)
    fig = plt.figure(figsize=(3, 3))
    plt.plot([i*x_step for i in range(k_range)], yerror, linewidth=4, color = "#009ACD")
    plt.scatter((best_k-1)*x_step, yerror[best_k-1], color = "red", s=150)
    plt.title('Elbow Method', fontdict={'fontsize': 15})
    plt.xlabel('k / Number of clusters', fontsize=10)
    plt.ylabel('# of uncertain pairs / all possible pairs', fontsize=10)
    if show:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_url), exist_ok=True)
        plt.savefig(save_url, bbox_inches="tight") 
    plt.close()
