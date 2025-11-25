import random
from math import sqrt
import threading

from queue import Queue
from multiprocessing import cpu_count

import numpy as np
from timeit import default_timer as timer
from sklearn.metrics import pairwise_distances


def lb_keogh(series_a, series_b, window_size):
    """Lower-bounding algorithm for DTW. For details please go here:
    https://www.cs.ucr.edu/~eamonn/LB_Keogh.htm
    
    Parameters
    ----------
    series_a : array_like
        The time series to compute the lower bound for.
    series_b : array_like
        The time series to compute the lower bound for.
    window_size : int
        The window size for DTW computation.
    
    Returns
    -------
    float :
        The lower bound.
    """
    lb_sum = 0
    for index, value in enumerate(series_a):

        # figure out windowing
        start_index = 0
        if index - window_size >= 0:
            start_index = index - window_size

        stop_index = index + window_size

        lower_bound = min(series_b[start_index:stop_index])
        upper_bound = max(series_b[start_index:stop_index])

        if value > upper_bound:
            lb_sum = lb_sum + (value - upper_bound) ** 2
        elif value < lower_bound:
            lb_sum = lb_sum + (value - lower_bound) ** 2

    return sqrt(lb_sum)


def dtw_distance(series_a, series_b, window_size):
    """Computes the DTW distance between two time series given a window
    size.
    
    Parameters
    ----------
    series_a : array_like
        The time series to compute the lower bound for.
    series_b : array_like
        The time series to compute the lower bound for.
    window_size : int
        The window size for DTW computation.
    
    Returns
    -------
    float :
        The DTW distance.
    """
    dtw = {}
    difference = abs(len(series_a) - len(series_b))
    w = max(window_size, difference)

    for i in range(-1, len(series_a)):
        for j in range(-1, len(series_b)):
            dtw[(i, j)] = float('inf')
    dtw[(-1, -1)] = 0

    for i in range(len(series_a)):
        for j in range(max(0, i - w), min(len(series_b), i + w)):
            dist = (series_a[i] - series_b[j]) ** 2
            dtw[(i, j)] = dist + min(dtw[(i-1, j)],dtw[(i, j-1)], dtw[(i-1, j-1)])

    return sqrt(dtw[len(series_a) - 1, len(series_b) - 1])


class DTWClustering(object):
    """Clusters a list of time series to form the desired number of clusters
    using DTW with LB_Keogh.
    
    Attributes
    ----------
    data : list
        The list of time series to cluster.
    k : int
        The desired number of clusters to compute.
    window_size : int
        The window size used to compute distances.
    max_iters : int, Default 10
        The maximum number of iterations to refine the clusters in.
    n_jobs : int, Default All cores
        The number of cpu cores to use.
    verbose : bool, Default True
        Flag to indicate if runtime output should be shown.
    centroids : array_like
        The time series centroids with index specifying the cluster group.
    clusters : dict
        The cluster assignments for the series. Key is the centroid index.
    """
    def __init__(self, data, k, max_iters=10, window_size=4,
        n_jobs = cpu_count(), verbose=True):
        self.data = data
        self.k = k
        self.n_jobs = n_jobs
        self.max_iters = max_iters
        self.window_size = window_size
        self.threads = []
        self.verbose = verbose
        self.queue = Queue()
        self.clusters = {}
        self.centroids = []
    
    def __compute_distance(self, series_index):
        """Computes the DTW distance for a given time series index."""
        series = self.data[series_index]
        minimum_distance = float('inf')
        closest_cluster = None

        for cluster_index, cluster_series in enumerate(self.centroids):
            lb_dist = lb_keogh(series, cluster_series, self.window_size)
            # Only compute DTW if LB_Keogh lower bound is promising
            if lb_dist < minimum_distance:
                current_distance = dtw_distance(series, cluster_series, self.window_size)
                if current_distance < minimum_distance:
                    minimum_distance = current_distance
                    closest_cluster = cluster_index

        # Fallback safeguard
        if closest_cluster is None:
            closest_cluster = 0

        if closest_cluster not in self.clusters:
            self.clusters[closest_cluster] = []

        self.clusters[closest_cluster].append(series_index)

    
    def __dequeue_worker(self):
        """Worker function for parallelism."""
        while True:
            item = self.queue.get()
            if item is None:
                break
            
            self.__compute_distance(item)
            self.queue.task_done()
        
    def __init_workers(self):
        """Create workers based on n_jobs."""
        for i in range(self.n_jobs):
            thread = threading.Thread(target=self.__dequeue_worker)
            thread.start()
            self.threads.append(thread)
            
    def __stop_workers(self):
        """Stops the worker threads."""
        for i in range(self.n_jobs):
            self.queue.put(None)

        for thread in self.threads:
            thread.join()
        
    def train(self):
        """Clusters the time series together."""
        np.random.seed(11)
        self.centroids = random.sample(list(self.data), self.k)        
        self.__init_workers()
        
        for iteration in range(self.max_iters):
            start = timer()
            self.clusters = {}
            
            for series_index, series in enumerate(self.data):                
                self.queue.put(series_index, False)
            
            if self.verbose: 
                print(timer() - start, 'queue placement complete')
            
            self.queue.join()
            
            if self.verbose:
                print(timer() - start, 'computations complete')
                
            #recalculate centroids of clusters
            for key in self.clusters:
                cluster_sum = 0
                for k in self.clusters[key]:
                    cluster_sum = cluster_sum + self.data[k]
                self.centroids[key] = [
                    m / len(self.clusters[key]) for m in cluster_sum
                ]
                
            if self.verbose:
                print(timer() - start, 'iteration complete')
        
        self.queue.join()
        self.__stop_workers()

    def predict(self, series):
        minimum_distance = float('inf')
        closest_cluster = None

        for cluster_index, cluster_series in enumerate(self.centroids):
            lb_dist = lb_keogh(series, cluster_series, self.window_size)
            if lb_dist < minimum_distance:
                current_distance = dtw_distance(series, cluster_series, self.window_size)
                if current_distance < minimum_distance:
                    minimum_distance = current_distance
                    closest_cluster = cluster_index

        if closest_cluster is None:
            closest_cluster = 0

        return (minimum_distance, closest_cluster)

class EuclideanClustering(object):
    """
    Clusters a list of time series using Euclidean distance.
    Structure mirrors DTWClustering for fair comparison.
    """

    def __init__(self, data, k, max_iters=10, n_jobs=cpu_count(), verbose=True):
        self.data = data
        self.k = k
        self.n_jobs = n_jobs
        self.max_iters = max_iters
        self.verbose = verbose
        self.threads = []
        self.queue = Queue()
        self.clusters = {}
        self.centroids = []

    def __compute_distance(self, series_index):
        """Compute Euclidean distance between a series and all centroids."""
        series = self.data[series_index]
        min_dist = float('inf')
        closest_cluster = None

        for cluster_index, centroid in enumerate(self.centroids):
            dist = sqrt(np.sum((series - centroid) ** 2))
            if dist < min_dist:
                min_dist = dist
                closest_cluster = cluster_index

        # fallback
        if closest_cluster is None:
            closest_cluster = 0

        if closest_cluster not in self.clusters:
            self.clusters[closest_cluster] = []

        self.clusters[closest_cluster].append(series_index)

    def __dequeue_worker(self):
        """Worker for parallel distance computation."""
        while True:
            item = self.queue.get()
            if item is None:
                break
            self.__compute_distance(item)
            self.queue.task_done()

    def __init_workers(self):
        for i in range(self.n_jobs):
            thread = threading.Thread(target=self.__dequeue_worker)
            thread.start()
            self.threads.append(thread)

    def __stop_workers(self):
        for i in range(self.n_jobs):
            self.queue.put(None)
        for thread in self.threads:
            thread.join()

    def train(self):
        """K-means style clustering using Euclidean distance."""
        np.random.seed(11)
        self.centroids = random.sample(list(self.data), self.k)
        self.__init_workers()

        for iteration in range(self.max_iters):
            start = timer()
            self.clusters = {}

            # Assign points to nearest centroid
            for series_index in range(len(self.data)):
                self.queue.put(series_index, False)

            if self.verbose:
                print(timer() - start, "queue placement complete")

            self.queue.join()

            if self.verbose:
                print(timer() - start, "computations complete")

            # Recompute centroids
            for key in self.clusters:
                cluster_sum = 0
                for idx in self.clusters[key]:
                    cluster_sum = cluster_sum + self.data[idx]
                self.centroids[key] = [
                    m / len(self.clusters[key]) for m in cluster_sum
                ]

            if self.verbose:
                print(timer() - start, "iteration complete")

        self.queue.join()
        self.__stop_workers()

    def predict(self, series):
        """Predict nearest Euclidean cluster for a new time series."""
        min_dist = float('inf')
        closest_cluster = None

        for cluster_index, centroid in enumerate(self.centroids):
            dist = sqrt(np.sum((series - centroid) ** 2))
            if dist < min_dist:
                min_dist = dist
                closest_cluster = cluster_index

        if closest_cluster is None:
            closest_cluster = 0

        return min_dist, closest_cluster

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def dtw_distance_matrix(data, window_size):
    """
    Compute a full pairwise DTW distance matrix.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_timestamps)
    window_size : int

    Returns
    -------
    D : ndarray, shape (n_samples, n_samples)
        Pairwise DTW distance matrix.
    """
    n = len(data)
    D = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            d = dtw_distance(data[i], data[j], window_size)
            D[i, j] = d
            D[j, i] = d

    return D


def silhouette_score_dtw(data, labels, window_size):
    """
    Compute silhouette score using DTW distance instead of Euclidean.

    Parameters
    ----------
    data : ndarray
        Time-series dataset.
    labels : array-like
        Cluster assignments.
    window_size : int

    Returns
    -------
    float : silhouette score (DTW version)
    """
    # silhouette is undefined if only 1 cluster OR clusters of size 1
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2 or np.any(counts <= 1):
        return None

    # compute DTW distance matrix
    D = dtw_distance_matrix(data, window_size)

    # compute silhouette score with precomputed DTW distances
    return silhouette_score(D, labels, metric="precomputed")

def evaluate_clustering(data, labels):
    """
    Compute standard clustering quality metrics.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
    labels : list or ndarray
        Cluster labels for each sample.

    Returns
    -------
    dict : metric → value
    """
    # Silhouette only valid if >= 2 clusters and no cluster of size 1
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) > 1 and np.all(counts > 1):
        sil = silhouette_score(data, labels)
    else:
        sil = None  # not defined

    return {
        "silhouette": sil,
        "davies_bouldin": davies_bouldin_score(data, labels),
        "calinski_harabasz": calinski_harabasz_score(data, labels)
    }

def labels_to_dict(labels):
    """
    Convert a label array into a dict: 
    cluster_id -> list of sample indices (string keys).
    """
    clusters = {}
    for idx, c in enumerate(labels):
        key = str(c)
        if key not in clusters:
            clusters[key] = []
        clusters[key].append(idx)
    return clusters


def benchmark_dtw_vs_euclidean(data, k=3, max_iters=10, window_size=4, verbose=True):
    """
    Runs both DTW and Euclidean clustering, compares results using standard metrics.

    Returns a dictionary containing:
        - cluster labels
        - centroids
        - clustering metrics
        - runtime

    Parameters
    ----------
    data : ndarray
        Time series dataset
    k : int
        Number of clusters
    max_iters : int
        Training iterations
    window_size : int
        DTW window size

    """

    results = {}

    # =========================
    # Run DTW Clustering
    # =========================
    if verbose:
        print("\n=== Running DTW Clustering ===")

    dtw_model = DTWClustering(
        data=data,
        k=k,
        max_iters=max_iters,
        window_size=window_size,
        verbose=verbose
    )

    start = timer()
    dtw_model.train()
    dtw_runtime = timer() - start

    # Extract labels
    dtw_labels = np.zeros(len(data), dtype=int)
    for cluster_id, members in dtw_model.clusters.items():
        dtw_labels[members] = cluster_id

    # Format DTW labels into requested dictionary format
    dtw_labels_dict = labels_to_dict(dtw_labels)

    results["DTW"] = {
        "labels": dtw_labels_dict,
        "centroids": dtw_model.centroids,
        "metrics": {
            "silhouette_dtw": silhouette_score_dtw(data, dtw_labels, window_size),
            "davies_bouldin": davies_bouldin_score(data, dtw_labels),
            "calinski_harabasz": calinski_harabasz_score(data, dtw_labels)
        },
        "runtime_sec": dtw_runtime
    }

    # =========================
    # Run Euclidean Clustering
    # =========================
    if verbose:
        print("\n=== Running Euclidean Clustering ===")

    euc_model = EuclideanClustering(
        data=data,
        k=k,
        max_iters=max_iters,
        verbose=verbose
    )

    start = timer()
    euc_model.train()
    euc_runtime = timer() - start

    euc_labels = np.zeros(len(data), dtype=int)
    for cluster_id, members in euc_model.clusters.items():
        euc_labels[members] = cluster_id

    euc_labels_dict = labels_to_dict(euc_labels)

    results["Euclidean"] = {
        "labels": euc_labels_dict,
        "centroids": euc_model.centroids,
        "metrics": evaluate_clustering(data, euc_labels),
        "runtime_sec": euc_runtime
    }

    return results