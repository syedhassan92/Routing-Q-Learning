"""
dijkstra_routing.py
====================
Computes the shortest (lowest-cost) path between two nodes in a weighted
graph using Dijkstra's algorithm via NetworkX.
"""

from typing import List, Tuple

import networkx as nx


def dijkstra_path(G: nx.Graph, source: int, target: int) -> Tuple[List[int], int]:
    """Find the shortest path and its total cost using Dijkstra's algorithm.

    Parameters
    ----------
    G : nx.Graph
        Weighted undirected graph.
    source : int
        Source node (router).
    target : int
        Destination node (router).

    Returns
    -------
    tuple[list[int], int]
        (path, cost) where *path* is the ordered list of nodes and *cost*
        is the sum of edge weights along that path.
    """
    path = nx.dijkstra_path(G, source, target, weight="weight")
    cost = nx.dijkstra_path_length(G, source, target, weight="weight")
    return path, int(cost)


def format_path(path: List[int]) -> str:
    """Return a human-readable arrow-separated path string.

    Parameters
    ----------
    path : list[int]
        Ordered list of node IDs.

    Returns
    -------
    str
        e.g. ``'0→1→4→7'``
    """
    return "->".join(str(n) for n in path)


if __name__ == "__main__":
    # Create a simple graph for testing since network_graph is missing
    G = nx.Graph()
    G.add_edges_from([
        (0, 1, {"weight": 10}), (1, 2, {"weight": 15}),
        (2, 3, {"weight": 10}), (3, 4, {"weight": 20}),
        (4, 5, {"weight": 10}), (5, 6, {"weight": 5}),
        (6, 7, {"weight": 15}), (0, 7, {"weight": 100}),
        (2, 7, {"weight": 50})
    ])

    path, cost = dijkstra_path(G, 0, 7)
    print(f"Dijkstra Path: [{format_path(path)}], Cost: {cost}")
