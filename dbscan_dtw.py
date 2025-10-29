import numpy as np
from math import sqrt
import numpy as np
from math import sqrt
from queue import Queue
from multiprocessing import cpu_count
import threading

# ---------- Distance Functions ----------

def lb_keogh(series_a, series_b, window_size):
    lb_sum = 0
    for index, value in enumerate(series_a):
        start_index = max(0, index - window_size)
        stop_index = min(len(series_b), index + window_size)
        lower_bound = min(series_b[start_index:stop_index])
        upper_bound = max(series_b[start_index:stop_index])

        if value > upper_bound:
            lb_sum += (value - upper_bound) ** 2
        elif value < lower_bound:
            lb_sum += (value - lower_bound) ** 2
    return sqrt(lb_sum)

def dtw_distance(series_a, series_b, window_size):
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
            dtw[(i, j)] = dist + min(dtw[(i-1, j)], dtw[(i, j-1)], dtw[(i-1, j-1)])

    return sqrt(dtw[len(series_a)-1, len(series_b)-1])

# --- Simple DTW Barycenter Averaging (DBA) ---
def dtw_barycenter_averaging(cluster_series, window_size=4, max_iters=10):
    """
    Compute a DTW barycenter (average) of a list of time series.
    """
    if len(cluster_series) == 1:
        return cluster_series[0]
    
    # Initialize barycenter as one random member
    barycenter = cluster_series[np.random.randint(len(cluster_series))].copy()
    
    for iteration in range(max_iters):
        associations = [[] for _ in range(len(barycenter))]
        
        for series in cluster_series:
            # Compute DTW alignment path
            dtw_matrix = np.full((len(barycenter)+1, len(series)+1), np.inf)
            dtw_matrix[0, 0] = 0
            for i in range(1, len(barycenter)+1):
                for j in range(max(1, i-window_size), min(len(series)+1, i+window_size)):
                    cost = (barycenter[i-1] - series[j-1])**2
                    dtw_matrix[i, j] = cost + min(
                        dtw_matrix[i-1, j],
                        dtw_matrix[i, j-1],
                        dtw_matrix[i-1, j-1]
                    )
            # Traceback alignment path
            i, j = len(barycenter), len(series)
            path = []
            while i > 0 and j > 0:
                path.append((i-1, j-1))
                steps = [dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]]
                move = np.argmin(steps)
                if move == 0:
                    i -= 1
                elif move == 1:
                    j -= 1
                else:
                    i -= 1
                    j -= 1
            path.reverse()

            # Aggregate aligned points
            for (a_i, b_j) in path:
                associations[a_i].append(series[b_j])

        # Update barycenter
        for i in range(len(barycenter)):
            if associations[i]:
                barycenter[i] = np.mean(associations[i])

    return barycenter

# ---------- DBSCAN with DTW ----------

class DTWDBSCAN:
    def __init__(self, data, eps=1.0, min_samples=3, window_size=4, verbose=True):
        """
        Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
        using DTW distance.
        """
        self.data = data
        self.eps = eps
        self.min_samples = min_samples
        self.window_size = window_size
        self.verbose = verbose
        self.labels_ = np.full(len(data), -1)
        self.cluster_id = 0
        self.cluster_representatives_ = {}

    def __get_neighbors(self, idx):
        """Return all points within eps DTW distance of point idx."""
        neighbors = []
        series_a = self.data[idx]
        for j, series_b in enumerate(self.data):
            if j == idx:
                continue
            if lb_keogh(series_a, series_b, self.window_size) > self.eps:
                continue
            dist = dtw_distance(series_a, series_b, self.window_size)
            if dist <= self.eps:
                neighbors.append(j)
        return neighbors

    def fit(self):
        """Run DBSCAN clustering."""
        visited = np.zeros(len(self.data), dtype=bool)

        for i in range(len(self.data)):
            if visited[i]:
                continue
            visited[i] = True

            neighbors = self.__get_neighbors(i)

            if len(neighbors) < self.min_samples:
                # Mark as noise
                self.labels_[i] = -1
                continue

            # Start new cluster
            if self.verbose:
                print(f"Starting cluster {self.cluster_id} with point {i}")
            self.labels_[i] = self.cluster_id

            seeds = neighbors.copy()
            while seeds:
                point = seeds.pop()
                if not visited[point]:
                    visited[point] = True
                    new_neighbors = self.__get_neighbors(point)
                    if len(new_neighbors) >= self.min_samples:
                        # Add new dense points
                        seeds.extend([n for n in new_neighbors if n not in seeds])

                if self.labels_[point] == -1:
                    self.labels_[point] = self.cluster_id

            self.cluster_id += 1

        if self.verbose:
            print("DBSCAN clustering complete.")
            print(f"Found {self.cluster_id} clusters.")

    def predict(self, series):
        """Find nearest cluster for a new series (optional)."""
        min_dist = float('inf')
        assigned_cluster = -1

        for cid in range(self.cluster_id):
            # pick one representative per cluster
            cluster_members = np.where(self.labels_ == cid)[0]
            centroid = np.mean([self.data[j] for j in cluster_members], axis=0)
            dist = dtw_distance(series, centroid, self.window_size)
            if dist < min_dist:
                min_dist = dist
                assigned_cluster = cid
        return assigned_cluster, min_dist

    def compute_cluster_representatives(self, method="dba", max_iters=10):
        """
        Compute representative time series for each cluster.
        method: "dba" (default) or "medoid"
        """
        self.cluster_representatives_ = {}

        for cid in range(self.cluster_id):
            members = [self.data[i] for i in np.where(self.labels_ == cid)[0]]
            if not members:
                continue

            if method == "medoid":
                # Choose the actual member with minimal total DTW distance
                dist_matrix = np.zeros((len(members), len(members)))
                for i in range(len(members)):
                    for j in range(i+1, len(members)):
                        d = dtw_distance(members[i], members[j], self.window_size)
                        dist_matrix[i, j] = dist_matrix[j, i] = d
                medoid_index = np.argmin(dist_matrix.sum(axis=1))
                self.cluster_representatives_[cid] = members[medoid_index]
            else:
                # DTW barycenter averaging
                self.cluster_representatives_[cid] = dtw_barycenter_averaging(
                    members, window_size=self.window_size, max_iters=max_iters
                )

        if self.verbose:
            print(f"Computed {len(self.cluster_representatives_)} cluster representatives using {method}.")
