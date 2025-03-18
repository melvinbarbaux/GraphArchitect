import networkx as nx
import numpy as np
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph, NearestNeighbors
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans


def build_knn_graph(data, k=5):
    """
    Builds a K-Nearest Neighbors (KNN) graph using a sparse neighborhood matrix.
    """
    A = kneighbors_graph(data, n_neighbors=k, mode="connectivity", include_self=False)
    return nx.from_scipy_sparse_matrix(A)


def build_epsilon_graph(data, epsilon=0.5):
    """
    Builds an epsilon graph by connecting points within a given radius.
    """
    A = radius_neighbors_graph(
        data, radius=epsilon, mode="connectivity", include_self=False
    )
    return nx.from_scipy_sparse_matrix(A)


def build_mst_graph(data):
    """
    Builds a Minimum Spanning Tree (MST) from the distance matrix.
    Note: For large datasets, constructing the full distance matrix can be expensive.
    """
    dist_matrix = squareform(pdist(data, metric="euclidean"))
    mst_sparse = minimum_spanning_tree(dist_matrix)
    return nx.from_scipy_sparse_matrix(mst_sparse)


def build_anchor_graph(data, num_anchors=50, k=3):
    """
    Builds an "anchor" graph by first clustering the data with KMeans,
    then connecting each point to its k nearest anchors.
    """
    # Using KMeans with explicit n_init for better robustness
    kmeans = KMeans(n_clusters=num_anchors, random_state=42, n_init="auto").fit(data)
    anchors = kmeans.cluster_centers_
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(anchors)
    _, anchor_indices = nbrs.kneighbors(data)

    # Create edges: each point is connected to its k nearest anchors
    edges = [
        (i, f"anchor_{a}")
        for i, anchor_ids in enumerate(anchor_indices)
        for a in anchor_ids
    ]
    G = nx.Graph()
    G.add_edges_from(edges)
    return G


def build_delaunay_graph(data):
    """
    Builds a graph based on Delaunay triangulation.
    Vectorized edge extraction with duplicate removal.
    """
    tri = Delaunay(data)
    edges = np.concatenate(
        [
            np.sort(tri.simplices[:, [0, 1]], axis=1),
            np.sort(tri.simplices[:, [0, 2]], axis=1),
            np.sort(tri.simplices[:, [1, 2]], axis=1),
        ],
        axis=0,
    )
    unique_edges = np.unique(edges, axis=0)
    return nx.from_edgelist(map(tuple, unique_edges))


def build_graph(method, data, **kwargs):
    """
    Builds a graph using the specified method.

    Parameters:
      - method: "knn", "epsilon", "mst", "anchor", or "delaunay"
      - data: data array
      - kwargs: method-specific parameters
    """
    if method == "knn":
        return build_knn_graph(data, k=kwargs.get("k", 5))
    elif method == "epsilon":
        return build_epsilon_graph(data, epsilon=kwargs.get("epsilon", 0.5))
    elif method == "mst":
        return build_mst_graph(data)
    elif method == "anchor":
        return build_anchor_graph(
            data, num_anchors=kwargs.get("num_anchors", 50), k=kwargs.get("anchor_k", 3)
        )
    elif method == "delaunay":
        return build_delaunay_graph(data)
    else:
        raise ValueError(f"Unknown method: {method}")
