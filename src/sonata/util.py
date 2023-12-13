import numpy as np
import scipy.sparse as sp 
import sklearn 
from sklearn.decomposition import PCA

from functools import wraps

# wrap two methods to avoid importing extra methods
def preserve_docstring(original_func):
    @wraps(original_func)
    def wrapper(*args, **kwargs):
        # Call the original function with all arguments and keyword arguments
        result = original_func(*args, **kwargs)
        return result
    return wrapper

#wrapped_normalize = preserve_docstring(sklearn.preprocessing.normalize)
def wrapped_normalize(X: np.ndarray, norm: str='l2', axis: int = 1) -> np.ndarray:
    """
    Normalize samples individually to unit norm.

    Parameters
    ----------
    X : np.ndarray
        The data array to be normalized.
    norm : str, optional
        The norm to use to normalize each non zero sample 
        (or each non-zero feature if axis is 0), options are 'l1', 'l2' and 'max', by default 'l2'.
    axis : int, optional
        Axis used to normalize the data along. If 1, independently normalize each sample, 
        otherwise (if 0) normalize each feature, by default 1.

    Returns
    -------
    X_normalized : np.ndarray
        Normalized input X.
    """
    return sklearn.preprocessing.normalize(X, norm, axis=axis)

def wrapped_pca(X: np.ndarray, n_components: int) -> np.ndarray:
    """
    Perform Principal Component Analysis (PCA) on the input data.

    Parameters
    ----------
    X : np.ndarray
        The input data to be transformed.
    
    n_components : int
        The number of components to keep. This determines the dimensionality of
        the transformed data.

    Returns
    -------
    np.ndarray
        The transformed data, where each row represents a sample and each column
        represents a principal component.

    """
    pca_instance = PCA(n_components=n_components).fit(X)
    X_pca = pca_instance.fit_transform(X)
    return X_pca

def load_data(matrix_file: str) -> np.ndarray:
    """
    Load data from various file formats and return as a NumPy array.

    Parameters
    ----------
    matrix_file : str
        The path to the input matrix file.

    Returns
    -------
    numpy.ndarray
        The loaded data as a NumPy array.

    Notes
    -----
    This function supports loading data from different file formats, including 'txt', 'csv', 'npz', and 'npy'.
    It automatically detects the file format based on the file extension and returns the data as a NumPy array.

    """
    file_type = matrix_file.split('.')[-1]
    if file_type == 'txt':
        data = np.loadtxt(matrix_file)
    elif file_type == 'csv':
        data = np.loadtxt(matrix_file, delimiter=',')
    elif file_type == 'npz':
        data = sp.load_npz(matrix_file)
    else:
        data = np.load(matrix_file) 

    # if file_type != 'npz':
        # print('data size={}'.format(data.shape))
    return data

def projection_barycentric(x: np.ndarray, y: np.ndarray, coupling: np.ndarray, XontoY: bool = True) -> tuple:
    """
    Perform barycentric projection from one domain to another.

    Parameters
    ----------
    x : numpy.ndarray
        The data points in the source domain.
    y : numpy.ndarray
        The data points in the target domain.
    coupling : numpy.ndarray
        The coupling matrix representing the relationship between domains.
    XontoY : bool, optional
        Flag indicating the direction of projection, by default True (X onto Y).

    Returns
    -------
    tuple
        A tuple containing two arrays (X_aligned and Y_aligned) representing the projected data in the target domain.

    Notes
    -----
    This function performs barycentric projection from one domain to another based on the coupling matrix.
    It can project the first domain onto the second domain (XontoY=True) or vice versa (XontoY=False).

    projection function from SCOT: https://github.com/rsinghlab/SCOT
    """
    if XontoY:
        #Projecting the first domain onto the second domain
        y_aligned=y
        weights=np.sum(coupling, axis = 0)
        X_aligned=np.matmul(coupling, y) / weights[:, None]
    else:
        #Projecting the second domain onto the first domain
        X_aligned = x
        weights=np.sum(coupling, axis = 0)
        y_aligned=np.matmul(np.transpose(coupling), x) / weights[:, None]

    return X_aligned, y_aligned