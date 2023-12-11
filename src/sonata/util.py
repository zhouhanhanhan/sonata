import numpy as np
import scipy.sparse as sp 
import sklearn 
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
def wrapped_normalize(X: np.ndarray, norm: str, axis: int = 1) -> np.ndarray:
    """
    Normalize samples individually to unit norm.

    Parameters
    ----------
    X : np.ndarray
        The data array to be normalized.
    norm : str
        The norm to use to normalize each non zero sample 
        (or each non-zero feature if axis is 0).
    axis : int
        Axis used to normalize the data along. If 1, independently normalize each sample, 
        otherwise (if 0) normalize each feature, by default 1.

    Returns
    -------
    X_normalized : np.ndarray
        Normalized input X.
    """
    return sklearn.preprocessing.normalize(X, norm, axis=axis)

from sklearn.decomposition import PCA as OriginalPCA
class Wrapped_PCA(OriginalPCA):
    __doc__ = OriginalPCA.__doc__
    def __init__(self, n_components: int):
        """Initialize the Wrapped PCA class."""
        super().__init__(n_components=n_components)
    def fit(self, X:np.ndarray, y:np.ndarray=None):
        """
        Fit the PCA model with the given data.

        Parameters
        ----------
        X : ndarray
            Input data.
        y : ndarray, optional
            Target data. Default is None.

        Returns
        -------
        self : object
            Fitted PCA model.
        """
        return super().fit(X, y)
    def fit_transform(self, X:np.ndarray, y:np.ndarray=None):
        """
        Fit the PCA model with the given data and apply dimensionality reduction.

        Parameters
        ----------
        X : ndarray
            Input data.
        y : ndarray, optional
            Target data. Default is None.

        Returns
        -------
        X_new : ndarray
            Transformed data.
        """
        return super().fit_transform(X, y)

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

def subsampling(data: np.ndarray, sample_size: int) -> np.ndarray:
    """
    Subsample data to a specified sample size.

    Parameters
    ----------
    data : numpy.ndarray
        The input data to be subsampled.
    sample_size : int
        The desired sample size.

    Returns
    -------
    numpy.ndarray
        The subsampled data.

    Notes
    -----
    This function subsamples the input data to the specified sample size.
    It uses linear spacing to select indices for subsampling and returns the subsampled data.

    """
    linspace = np.linspace(0, data.shape[0] - 1, sample_size, dtype= int)
    data_new = data[linspace]
    return data_new