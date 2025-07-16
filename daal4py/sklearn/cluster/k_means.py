# ==============================================================================
# Copyright 2014 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import scipy.sparse as sp
import daal4py

from .._utils import getFPType

def _validate_center_shape(X, n_clusters, centers):
    """Check if centers is compatible with X and n_clusters."""
    if centers.shape[0] != n_clusters:
        raise ValueError(
            f"The shape of the initial centers {centers.shape} does not "
            f"match the number of clusters {n_clusters}."
        )
    if centers.shape[1] != X.shape[1]:
        raise ValueError(
            f"The shape of the initial centers {centers.shape} does not "
            f"match the number of features in the data {X.shape[1]}."
        )

def _daal4py_compute_starting_centroids(X, n_clusters, init_method, random_state):
    """
    Compute initial centroids using oneDAL.
    """
    X_fptype = getFPType(X)
    is_sparse = sp.issparse(X)

    def _seed():
        return random_state.randint(np.iinfo("i").max)

    if isinstance(init_method, str):
        if init_method == "k-means++":
            method = "plusPlusCSR" if is_sparse else "plusPlusDense"
            engine = daal4py.engines_mt19937(fptype=X_fptype, seed=_seed())
            kmeans_init = daal4py.kmeans_init(
                nClusters=n_clusters,
                fptype=X_fptype,
                method=method,
                nTrials=2 + int(np.log(n_clusters)),
                engine=engine,
            )
            centroids = kmeans_init.compute(X).centroids
        elif init_method == "random":
            method = "randomCSR" if is_sparse else "randomDense"
            engine = daal4py.engines_mt19937(fptype=X_fptype, seed=_seed())
            kmeans_init = daal4py.kmeans_init(
                nClusters=n_clusters,
                fptype=X_fptype,
                method=method,
                engine=engine,
            )
            centroids = kmeans_init.compute(X).centroids
        elif init_method == "deterministic":
            method = "lloydCSR" if is_sparse else "defaultDense"
            kmeans_init = daal4py.kmeans_init(
                nClusters=n_clusters,
                fptype=X_fptype,
                method=method,
            )
            centroids = kmeans_init.compute(X).centroids
        else:
            raise ValueError(
                f"Unknown init method: {init_method}"
            )
    elif hasattr(init_method, "__array__"):
        centroids = np.ascontiguousarray(init_method, dtype=X.dtype)
        _validate_center_shape(X, n_clusters, centroids)
    elif callable(init_method):
        centroids = init_method(X, n_clusters, random_state)
        centroids = np.ascontiguousarray(centroids, dtype=X.dtype)
        _validate_center_shape(X, n_clusters, centroids)
    else:
        raise ValueError(
            f"init should be a string, ndarray, or callable, got {type(init_method)}"
        )

    return centroids

def _daal4py_kmeans_fit(
    X,
    n_clusters,
    max_iter,
    tol,
    init,
    n_init,
    verbose,
    random_state,
):
    """
    Run oneDAL KMeans clustering.
    """

    X_fptype = getFPType(X)
    is_sparse = sp.issparse(X)
    method = "lloydCSR" if is_sparse else "defaultDense"

    abs_tol = tol * np.var(X, axis=0).mean() if tol > 0 else 0.0

    best_inertia = None
    best_centers = None
    best_labels = None
    best_n_iter = None

    daal_kmeans = daal4py.kmeans(
        nClusters=n_clusters,
        maxIterations=max_iter,
        fptype=X_fptype,
        accuracyThreshold=abs_tol,
        resultsToEvaluate="computeCentroids",
        method=method,
    )

    for i in range(n_init):
        centroids = _daal4py_compute_starting_centroids(
            X, n_clusters, init, random_state
        )
        result = daal_kmeans.compute(X, centroids)

        inertia = result.objectiveFunction[0, 0]
        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia
            best_centers = result.centroids.copy()
            best_n_iter = int(result.nIterations[0, 0])

    labels, final_inertia = _daal4py_kmeans_predict(
        X, n_clusters, best_centers
    )

    return best_centers, labels, final_inertia, best_n_iter

def _daal4py_kmeans_predict(X, n_clusters, centroids):
    """
    Predict labels using DAAL.
    """
    X_fptype = getFPType(X)
    is_sparse = sp.issparse(X)
    method = "lloydCSR" if is_sparse else "defaultDense"

    kmeans_algo = daal4py.kmeans(
        nClusters=n_clusters,
        maxIterations=0,
        fptype=X_fptype,
        method=method,
        resultsToEvaluate="computeAssignments|computeExactObjectiveFunction",
    )

    res = kmeans_algo.compute(X, centroids)
    labels = res.assignments[:, 0]
    inertia = res.objectiveFunction[0, 0]

    return labels, inertia
