"""
Hierarchical surrogate construction: ER / configuration / SBM per layer.

The surrogate starts with all original nodes and is assembled layer by
layer in component order (backbone first).
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Hashable, Sequence

import networkx as nx
import numpy as np

from datasets import DISTANCE_KEY
from layers import Edge, LayerAssignment
from mixture import GammaMixture

log = logging.getLogger(__name__)

#: Edge attribute recording the layer at which a surrogate edge was generated.
LAYER_KEY = "layer"

GENERATORS = ("er",) #we may try other that Erdos-Renyi


def _ekey(u: Hashable, v: Hashable) -> Edge:
    """Canonical undirected edge key (stable across processes)."""
    return (u, v) if str(u) <= str(v) else (v, u)


def _sample_distinct_pairs(a_nodes: Sequence, b_nodes: Sequence, m: int, taken: set, rng: np.random.Generator, same_set: bool) -> list[Edge]:
    """
    Sample `m` distinct unordered node pairs (one endpoint from each list, no self-loops) that are not in `taken`. Sampled keys are added to `taken`.
    Falls back to exhaustive enumeration when rejection sampling stalls (dense regime); may return fewer than `m` pairs if the pair space is exhausted.
    """
    out: list[Edge] = []
    if m <= 0:
        return out
    n_a, n_b = len(a_nodes), len(b_nodes)
    if n_a == 0 or n_b == 0 or (same_set and n_a < 2):
        return out

    attempts, limit = 0, 30 * m + 1000
    log.debug("Fast rejection sampling")
    while len(out) < m and attempts < limit:
        attempts += 1
        u = a_nodes[rng.integers(n_a)]
        v = b_nodes[rng.integers(n_b)]
        if u == v:
            continue
        key = _ekey(u, v)
        if key in taken:
            continue
        taken.add(key)
        out.append(key)

    if len(out) < m:
        # Dense regime: enumerate the remaining free pairs and sample exactly.
        log.debug("Dense regime, exact enumeration")
        if same_set:
            pool = [_ekey(a_nodes[i], a_nodes[j]) for i in range(n_a) for j in range(i + 1, n_a)]
        else:
            pool = [_ekey(u, v) for u in a_nodes for v in b_nodes if u != v]
        pool = list(set(pool) - taken)
        pool.sort(key=lambda e: (str(e[0]), str(e[1])))
        take = min(m - len(out), len(pool))
        for i in rng.choice(len(pool), size=take, replace=False) if take else []:
            key = pool[int(i)]
            taken.add(key)
            out.append(key)
    return out


def _sample_er_layer(allowed: Sequence, m: int, taken: set, rng: np.random.Generator) -> list[Edge]:
    """G(n, m) on the allowed node set, avoiding already-present edges."""
    return _sample_distinct_pairs(allowed, allowed, m, taken, rng, same_set=True)


def _sample_configuration_layer(original_edges: Sequence[Edge], taken: set, rng: np.random.Generator) -> tuple[list[Edge], int]:
    """
    Configuration model matching the degree sequence of the original layer
    subgraph, projected to a simple graph. Returns ``(edges, n_lost)`` where
    losses come from self-loops, multi-edges and collisions with existing
    surrogate edges.
    """
    pass


def _sample_sbm_layer(original_edges: Sequence[Edge], taken: set, rng: np.random.Generator) -> tuple[list[Edge], list[dict]]:
    """
    Degree-corrected-free SBM layer: blocks are the layer at which nodes
    first appear. The block-pair edge counts of the original layer are
    used directly (i.e. we sample conditionally on the estimated counts,
    which matches both the estimated probabilities in expectation and the
    layer's edge budget exactly).
    """
    pass


def build_surrogate(G: nx.Graph, assignment: LayerAssignment, generator: str, rng: np.random.Generator) -> tuple[nx.Graph, dict]:
    """
    Assemble one surrogate graph layer by layer (backbone first).

    Returns ``(H, report)``. Every generated edge carries the layer rank it
    was generated at under the `LAYER_KEY` attribute.
    """
    if generator not in GENERATORS:
        raise ValueError(f"Unknown generator {generator}; expected one of {GENERATORS}.")
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    taken: set = set()
    cumulative = assignment.cumulative_node_sets()
    report: dict = {"generator": generator, "layers": []}
    new = []

    for k in range(1, assignment.n_layers + 1):
        original = assignment.layer_edges(k)
        m = len(original)
        allowed = sorted(cumulative[k - 1], key=str)
        entry: dict = {"rank": k, "target": m, "generated": 0, "lost": m}

        if m == 0 or len(allowed) < 2:
            if m:
                log.warning("Layer %d: %d edges requested but only %d allowed nodes; skipping.", k, m, len(allowed))
            report["layers"].append(entry)
            continue

        if generator == "er":
            new = _sample_er_layer(allowed, m, taken, rng)

        if len(new) < m:
            log.warning("Layer %d (%s): pair space exhausted, generated %d/%d edges.", k, generator, len(new), m)
        for u, v in new:
            H.add_edge(u, v, **{LAYER_KEY: k})
        entry["generated"] = len(new)
        entry["lost"] = m - len(new)
        report["layers"].append(entry)
    return H, report


def assign_surrogate_distances(H: nx.Graph, assignment: LayerAssignment, model: GammaMixture, rng: np.random.Generator, feature_attr: str = DISTANCE_KEY) -> None:
    """
    Give each surrogate edge a distance sampled from the Gamma component of the layer it was generated at (in place). This makes the surrogate
    comparable to the original under the raw-distance filtration.
    """
    comp_of_rank = assignment.component_order
    for _, _, data in H.edges(data=True):
        c = int(comp_of_rank[int(data[LAYER_KEY]) - 1])
        data[feature_attr] = float(rng.gamma(model.alphas_[c], 1.0/model.betas_[c]))


def surrogate_edge_layers(H: nx.Graph) -> tuple[list[Edge], list[int]]:
    """Edges of a surrogate together with the layer each was generated at."""
    edges: list[Edge] = []
    ranks: list[int] = []
    for u, v, data in H.edges(data=True):
        edges.append((u, v))
        ranks.append(int(data[LAYER_KEY]))
    return edges, ranks
