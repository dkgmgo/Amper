"""
Component ordering, edge-layer extraction and structural diagnostics.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Hashable

import networkx as nx
import numpy as np

from datasets import DISTANCE_KEY
from mixture import GammaMixture

log = logging.getLogger(__name__)

Edge = tuple[Hashable, Hashable]


def edge_distances(G: nx.Graph, feature_attr: str = DISTANCE_KEY) -> tuple[list[Edge], np.ndarray]:
    """
    Collect edges with a valid (finite, non-negative) distance value.
    """
    edges: list[Edge] = []
    dists: list[float] = []
    n_bad = 0
    for u, v, data in G.edges(data=True):
        try:
            d = float(data.get(feature_attr))
        except (TypeError, ValueError):
            d = math.nan
        if math.isfinite(d) and d >= 0.0:
            edges.append((u, v))
            dists.append(d)
        else:
            n_bad += 1
    if n_bad:
        log.warning("Dropping %d edges without a valid '%s' attribute.", n_bad, feature_attr)
    if not edges:
        raise ValueError(f"No edge has a valid '{feature_attr}' attribute.")
    return edges, np.asarray(dists, dtype=float)


@dataclass
class LayerAssignment:
    """Per-edge layer assignment derived from a fitted Gamma mixture."""

    edges: list[Edge]
    distances: np.ndarray
    hard_ranks: np.ndarray
    responsibilities: np.ndarray
    component_order: np.ndarray
    means: np.ndarray
    weights: np.ndarray

    @property
    def n_layers(self) -> int:
        return int(len(self.component_order))

    def layer_edges(self, rank: int) -> list[Edge]:
        """Original edges hard-assigned to layer `rank` (1-based)."""
        idx = np.nonzero(self.hard_ranks == rank)[0]
        return [self.edges[i] for i in idx]

    def layer_nodes(self, rank: int) -> set:
        """Nodes incident to at least one edge hard-assigned to layer `rank` (1-based). A node can belong to several layers."""
        nodes: set = set()
        for u, v in self.layer_edges(rank):
            nodes.add(u)
            nodes.add(v)
        return nodes

    def node_first_layer(self) -> dict[Hashable, int]:
        """Rank of the earliest layer each node appears in."""
        first: dict[Hashable, int] = {}
        for (u, v), r in zip(self.edges, self.hard_ranks):
            r = int(r)
            for w in (u, v):
                if w not in first or r < first[w]:
                    first[w] = r
        return first


def assign_layers(G: nx.Graph, model: GammaMixture, feature_attr: str = DISTANCE_KEY) -> LayerAssignment:
    """
    Assign each edge of `G` to a layer using the fitted mixture. Components are re-ordered by decreasing mean so that rank 1 is the
    backbone; responsibilities columns follow the same rank order.
    """
    edges, x = edge_distances(G, feature_attr)
    order = np.argsort(-model.means_, kind="stable")
    resp = model.predict_proba(x)[:, order]
    hard = resp.argmax(axis=1).astype(int) + 1
    if model.n_components == 1:
        log.warning("Single mixture component: simple hierarchy with one layer.")
    return LayerAssignment(
        edges=edges,
        distances=x,
        hard_ranks=hard,
        responsibilities=resp,
        component_order=order,
        means=model.means_[order],
        weights=model.weights_[order],
    )


def layer_diagnostics(assignment: LayerAssignment) -> list[dict]:
    """
    Structural sanity checks per layer (a 1D mixture is structure-blind).

    Returns one dict per layer with edge/node counts, connected component
    count and density of the induced subgraph.
    """
    rows: list[dict] = []
    for k in range(1, assignment.n_layers + 1):
        edges = assignment.layer_edges(k)
        nodes: set = set()
        for u, v in edges:
            nodes.add(u)
            nodes.add(v)
        sub = nx.Graph()
        sub.add_nodes_from(nodes)
        sub.add_edges_from(edges)
        n, m = len(nodes), len(edges)
        rows.append({
            "rank": k,
            "weight": float(assignment.weights[k - 1]),
            "mean_distance": float(assignment.means[k - 1]),
            "n_edges": m,
            "n_nodes": n,
            "n_components": nx.number_connected_components(sub) if n else 0,
            "density": (2.0 * m / (n * (n - 1))) if n > 1 else 0.0,
        })
        if m == 0:
            log.warning("Layer %d is empty (no edge hard-assigned to it).", k)
    return rows
