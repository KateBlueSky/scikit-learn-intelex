import numpy as np
import numbers
from .._utils import PatchingConditionsChain
from sklearn.utils.validation import _check_sample_weight, _num_samples
from scipy.sparse import issparse

def check_kmeans_patching_conditions(estimator, X, sample_weight):
    """
    Returns True if DAAL patching can be used.
    """
    class_name = estimator.__class__.__name__
    patching_status = PatchingConditionsChain(
        f"sklearn.cluster.{class_name}.fit"
    )
    
    n_samples = _num_samples(X)
    
    # check whether n_clusters < n_samples
    correct_count = estimator.n_clusters < n_samples
    
    is_data_supported = (
        hasattr(estimator, "_onedal_estimator")
        and estimator._onedal_estimator is not None
        and estimator._onedal_estimator._onedal_version >= (2024, "P", 700)
        and getattr(X, "getformat", lambda: None)() == "csr"
    ) or not issparse(X)
    
    acceptable_sample_weights = True
    if sample_weight is not None:
        if isinstance(sample_weight, numbers.Number) or isinstance(sample_weight, str):
            acceptable_sample_weights = True
        else:
            sample_weight = _check_sample_weight(
                sample_weight,
                X,
                dtype=X.dtype if hasattr(X, "dtype") else None,
            )
            acceptable_sample_weights = np.all(sample_weight == sample_weight[0])
    
    patching_status.and_conditions(
        [
            (
                estimator.algorithm in ["auto", "full", "lloyd", "elkan"],
                "Only 'lloyd' algorithm is supported; 'elkan' runs using lloyd.",
            ),
            (estimator.n_clusters != 1, "n_clusters=1 is not supported"),
            (correct_count, "n_clusters must be smaller than number of samples."),
            (
                acceptable_sample_weights,
                "oneDAL only supports None, constant, or equal sample weights.",
            ),
            (
                is_data_supported,
                "Supported formats: Dense or CSR (oneDAL version >= 2024.7.0).",
            ),
        ]
    )
    
    patching_status.write_log()
    return patching_status.is_success
