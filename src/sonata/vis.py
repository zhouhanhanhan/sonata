import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import typing
from . import util


def plt_domain_by_labels(data_mat: np.ndarray, label: np.ndarray, color: str = '#009ACD', title: str = None, y_tick_labels: list = None, save_url: str = '', a: float = 0.8, show: bool = True) -> None:
    """
    Scatter plot of a single data domain with labels.
    Subplot1 : unlabeled data scatters   subplot2 : data scatters colored by labels.

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
    Scatter plot of aligned data from two domains with labels.

    Parameters
    ----------
    X_new : np.ndarray
        Data from the first domain.
    y_new : np.ndarray
        Data from the second domain.
    label1 : np.ndarray
        Labels for data in the first domain.
    label2 : np.ndarray
        Labels for data in the second domain.
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
    ambiguous_cell_groups: dict, 
    ambiguous_links: list, 
    y_tick_labels: str = None, 
    save_url: str = None, 
    cl_alpha: float = 0.1, 
    curve_style : bool = False,
    show=True
) -> None:
    """
    Scatter plot of data with all cell-cell ambiguities. This function helps to visualize all cell ambiguities.

    Parameters
    ----------
    data_mat : np.ndarray
        The data matrix containing data points.
    label : np.ndarray
        The labels associated with each data point.
    ambiguous_cell_groups : dict
        A dictionary of ambiguous groups, where key is the group label and 
        value is an array containing indices of nodes in an ambiguous group.
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
    if len(ambiguous_cell_groups) > 1:
        ambiguous_nodes = np.concatenate(list(ambiguous_cell_groups.values()))
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


def plt_ambiguous_groups(
    data_mat: np.ndarray,
    ambiguous_cell_groups: dict,
    save_url: str = '',
    alpha: float = 0.8,
    show: bool = True,
) -> None:
    """
    Visualize mutually ambiguous groups identified by SONATA based on labels.

    Parameters
    ----------
    data_mat : np.ndarray
        Data points.
    ambiguous_cell_groups : dict
        A dictionary of ambiguous groups, where key is the group label and 
        value is an array containing indices of nodes in an ambiguous group.
    save_url : str, optional
        The path to save the plot as an image file, by default ''.
    alpha : float, optional
        Transparency value for data points, by default 0.8.
    show : bool, optional
        Whether to display the plot, by default True.

    Returns
    -------
    None
    """
    pca = PCA(n_components=2).fit(data_mat)
    data_embed = pca.fit_transform(data_mat)
    
    ambiguous_nodes = np.asarray([], dtype=int)
    ambiguous_labels = np.asarray([], dtype=int)
    for label, nodes in ambiguous_cell_groups.items():
        ambiguous_nodes = np.concatenate([ambiguous_nodes, nodes])
        ambiguous_labels = np.concatenate([ambiguous_labels, np.asarray([label]*len(nodes))])
    assert len(ambiguous_nodes) == len(ambiguous_labels)
    unambuguous_nodes = np.setdiff1d(list(range(data_mat.shape[0])), ambiguous_nodes)

    data_pca_ambiguous = data_embed[ambiguous_nodes, :]

    fig = plt.figure(figsize=(6, 6))
    plt.scatter(list(data_embed[unambuguous_nodes, 0]), list(data_embed[unambuguous_nodes, 1]), 
                c="grey", alpha=alpha, label='Certain', s=25)

    if len(ambiguous_cell_groups) > 1:    
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


def plt_alternaltive_mappings_by_label(
    data1: np.ndarray, 
    data2: np.ndarray, 
    label1: np.ndarray, 
    label2: np.ndarray, 
    map_mat_aligner: typing.Generator[np.ndarray, None, None], 
    max_return: int=4,
    title1: str = "Domain 1",
    title2: str = "Domain 2",
    y_tick_labels: list = None) -> None:
    """
    Visualize alternative mappings generated by SONATA between two datasets based on labels.

    Parameters
    ----------
    data1 : np.ndarray
        The first dataset.
    data2 : np.ndarray
        The second dataset.
    label1 : np.ndarray
        Labels corresponding to the first dataset.
    label2 : np.ndarray
        Labels corresponding to the second dataset.
    map_mat_aligner : typing.Generator[np.ndarray, None, None]
        A generator that yields alignment mappings.
    max_return : int, optional
        The maximum number of alternative mappings to plot. Set max_return=-1 to visualize all alternaltive solutions, byefault is 4.
    title1 : str, optional
        Title for the first dataset, by default 'Domain 1'.
    title2 : str, optional
        Title for the second dataset, by default 'Domain 2'.
    y_tick_labels : list, optional
        Labels for color bar ticks, by default None.

    Returns
    -------
    None

    """   
    for idx, mapping in enumerate(map_mat_aligner, start=1):
        if max_return != -1 and idx > max_return:
            continue
        x_aligned, y_aligned = util.projection_barycentric(data1, data2, mapping)
        plt_mapping_by_labels(x_aligned, y_aligned, label1, label2, title1=title1, title2=title2, y_tick_labels=y_tick_labels)
    print("Sonata finds {} alternative solutions. Plotting {} of them. If you want to visualize all alternaltive solutions, set max_return=-1.".format(
        idx, idx if idx < max_return else max_return))  