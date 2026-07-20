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
from gudhi.representations import SlicedWassersteinDistance

from layers import Edge

log = logging.getLogger(__name__)

Diagram = np.ndarray
Diagrams = list[Diagram]

#: Wasserstein variants selectable from the pipeline config.
WASSERSTEIN_METHODS = ("sliced", "exact")

#: Above this many bars the exact transport matrix is large enough to be worth a warning.
_EXACT_WASSERSTEIN_WARN_BARS = 8000


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
    Exact Wasserstein distance between two diagrams.
    """
    n = max(len(_as_diagram(d1)), len(_as_diagram(d2)))
    if n > _EXACT_WASSERSTEIN_WARN_BARS:
        log.warning("Exact Wasserstein on %d bars: the transport matrix a lot of memory. Consider method='sliced'.", n)
    return _gudhi_wasserstein(d1, d2, order=order, internal_p=float('inf'))


def sliced_wasserstein_distance(d1: Diagram, d2: Diagram, num_directions: int = 50) -> float:
    """
    Sliced Wasserstein distance: project both diagrams onto `num_directions`
    lines and average the 1-D Wasserstein distances. Linear in memory rather than quadratic.
    """
    a, b = _as_diagram(d1), _as_diagram(d2)
    if a.size == 0 and b.size == 0:
        return 0.0
    return float(SlicedWassersteinDistance(num_directions=num_directions).fit([a]).transform([b])[0][0])


def _as_diagram(d: Optional[Diagram]) -> Diagram:
    """Coerce a possibly empty or 1-D persistence array to shape (n, 2)."""
    a = np.asarray(d, dtype=float) if d is not None else np.empty((0, 2))
    return a.reshape(-1, 2) if a.size else np.empty((0, 2))


def _finite_max(diagrams: Sequence[Diagram]) -> float:
    """Largest finite birth/death across the given diagrams, or nan if none."""
    vals = [a[np.isfinite(a)] for a in diagrams if a.size]
    vals = [v for v in vals if v.size]
    return float(np.max(np.concatenate(vals))) if vals else float("nan")


def cap_diagram(d: Diagram, cap: float) -> Diagram:
    """Replace infinite deaths with `cap` so essential bars stay comparable."""
    a = _as_diagram(d).copy()
    if a.size:
        a[~np.isfinite(a[:, 1]), 1] = cap
    return a


def diagram_distances(original: Diagrams, surrogate: Diagrams, dims: Sequence[int] = (0, 1), cap: Optional[float] = None, wasserstein_method: str = "sliced", num_directions: int = 50) -> dict[tuple[int, str], float]:
    """
    All (dimension, metric) distances between two diagram sets.
    Essential bars (infinite death) are capped at `cap` before comparison.
    """
    if wasserstein_method not in WASSERSTEIN_METHODS:
        raise ValueError(f"Unknown wasserstein_method {wasserstein_method}; expected one of {WASSERSTEIN_METHODS}.")

    orig = [_as_diagram(original[dim]) if original else _as_diagram(None) for dim in dims]
    surr = [_as_diagram(surrogate[dim]) if surrogate else _as_diagram(None) for dim in dims]

    if cap is None:
        cap = _finite_max(orig + surr)
        if not math.isfinite(cap):
            cap = 1.0
            log.warning("No finite birth/death in either diagram; capping essential bars at %g.", cap)
        log.debug("Capping essential bars at %g (derived from diagrams).", cap)

    out: dict[tuple[int, str], float] = {}
    for dim, a, b in zip(dims, orig, surr):
        a, b = cap_diagram(a, cap), cap_diagram(b, cap)
        out[(dim, "bottleneck")] = bottleneck_distance(a, b)
        if dim > 0 and wasserstein_method == "sliced":
            out[(dim, "sliced_wasserstein")] = sliced_wasserstein_distance(a, b, num_directions=num_directions)
        else:
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
