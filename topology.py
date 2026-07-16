"""
Filtration, persistent homology and diagram distances
"""

from __future__ import annotations

import logging
import math
from typing import Hashable, Mapping, Optional, Sequence

import gudhi as gd
import networkx as nx
import numpy as np
from gudhi.wasserstein import wasserstein_distance as _gudhi_wasserstein

from layers import Edge

log = logging.getLogger(__name__)

Diagram = np.ndarray
Diagrams = list[Diagram]


def build_simplex_tree(edges: Sequence[Edge], edge_values: Sequence[float], expansion_dim: int = 2) -> gd.SimplexTree:
    """
    Persistence diagrams of the clique complex of a filtered graph.
    """
    st = gd.SimplexTree()
    nodes = set([u for e in edges for u in e])
    index = {v: i for i, v in enumerate(nodes)}
    for v in nodes:
        st.insert([index[v]], filtration=0.0)
    for (u, v), f in zip(edges, edge_values):
        st.insert([index[u], index[v]], filtration=float(f))
    
    st.expansion(expansion_dim)
    st.compute_persistence()
    return st


def bottleneck_distance(d1: Diagram, d2: Diagram) -> float:
    """
    Bottleneck distance between two diagrams.
    """
    return gd.bottleneck_distance(d1, d2)


def wasserstein_distance(d1: Diagram, d2: Diagram, order: float = 1.0) -> float:
    """
    Wasserstein distance between two diagrams.
    """
    return _gudhi_wasserstein(d1, d2, order=order, internal_p=float('inf'), keep_essential_parts=True)


def diagram_distances(original: Diagrams, surrogate: Diagrams, dims: Sequence[int] = (0, 1)) -> dict[tuple[int, str], float]:
    """
    All (dimension, metric) distances between two diagram sets.
    """
    out: dict[tuple[int, str], float] = {}
    empty = np.empty((0, 2))
    for dim in dims:
        a = original[dim] if original else empty
        b = surrogate[dim] if surrogate else empty
        out[(dim, "bottleneck")] = bottleneck_distance(a, b)
        out[(dim, "wasserstein")] = wasserstein_distance(a, b)
    return out


def get_topology(edgs: list, dists: list, expansion_dim: int) -> Diagrams:
    pairs = list(zip(edgs, dists))
    pairs = sorted(pairs, key= lambda p: p[1])
    edges, distances = [], []
    for e, d in pairs:
        edges.append(e)
        distances.append(d)
    st = build_simplex_tree(edges, distances, expansion_dim)
    return [st.persistence_intervals_in_dimension(i) for i in (0,1)]
