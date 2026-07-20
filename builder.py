"""
Hierarchical surrogate construction: ER / ER stratified.

The surrogate starts with all original nodes and is assembled layer by layer.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Hashable, Mapping, Sequence

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans

from datasets import DISTANCE_KEY
from layers import Edge, LayerAssignment
from mixture import GammaMixture

log = logging.getLogger(__name__)

#: Edge attribute recording the layer at which a surrogate edge was generated.
LAYER_KEY = "layer"

GENERATORS = ("er", "er_strat")


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


def _node_responsibility_profiles(assignment: LayerAssignment, weight_by_distance: bool = False) -> tuple[list[Hashable], np.ndarray]:
    """For each node, average the responsibility vectors of its incident edges."""
    K = assignment.responsibilities.shape[1]
    accum: dict[Hashable, np.ndarray] = defaultdict(lambda: np.zeros(K))
    weight: dict[Hashable, float] = defaultdict(float)

    for idx, (u, v) in enumerate(assignment.edges):
        r = assignment.responsibilities[idx]
        w = assignment.distances[idx] if weight_by_distance else 1.0
        for n in (u, v):
            accum[n] += r * w
            weight[n] += w

    nodes = sorted(accum, key=str)
    profiles = np.vstack([accum[n] / weight[n] for n in nodes])
    return nodes, profiles


def _cluster_nodes_by_responsibility(assignment: LayerAssignment, rng: np.random.Generator, n_clusters: int = 2, weight_by_distance: bool = False) -> dict[Hashable, int]:
    """KMeans on per-node responsibility profiles."""
    nodes, profiles = _node_responsibility_profiles(assignment, weight_by_distance=weight_by_distance)
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=rng.integers(0, 2**32)).fit_predict(profiles)

    counts = Counter(labels)
    log.info("Responsibility KMeans (%d clusters): %s", n_clusters, dict(sorted(counts.items())))
    return {n: int(labels[i]) for i, n in enumerate(nodes)}


def _block_pair_budgets(original_edges: Sequence[Edge], layer_nodes: set, node_blocks: Mapping[Hashable, int]) -> tuple[dict[tuple[int, int], int], dict[int, list]]:
    """Block-pair edge counts of the original layer"""
    nodes_by_block: dict[int, list] = defaultdict(list)
    for n in sorted(layer_nodes, key=str):
        nodes_by_block[node_blocks[n]].append(n)

    budgets: dict[tuple[int, int], int] = defaultdict(int)
    for u, v in original_edges:
        ba, bb = node_blocks[u], node_blocks[v]
        budgets[(ba, bb) if ba <= bb else (bb, ba)] += 1
    return dict(budgets), nodes_by_block


def _sample_er_layer(allowed: Sequence, m: int, taken: set, rng: np.random.Generator) -> list[Edge]:
    """G(n, m) on the allowed node set, avoiding already-present edges."""
    return _sample_distinct_pairs(allowed, allowed, m, taken, rng, same_set=True)


def _sample_er_stratified_layer(original_edges: Sequence[Edge], layer_nodes: set, node_blocks: Mapping[Hashable, int], taken: set, rng: np.random.Generator) -> tuple[list[Edge], dict]:
    """
    Stratified G(n, m): one independent G(n, m) per block pair, each matching the
    original layer's edge count for that pair.
    """
    budgets, nodes_by_block = _block_pair_budgets(original_edges, layer_nodes, node_blocks)

    edges: list[Edge] = []
    strata: dict = {}
    for (ba, bb), m in sorted(budgets.items()):
        new = _sample_distinct_pairs(nodes_by_block[ba], nodes_by_block[bb], m, taken, rng, same_set=(ba == bb))
        label = f"{ba}-{bb}"
        if len(new) < m:
            log.warning("Stratum %s: generated %d/%d edges.", label, len(new), m)
        edges.extend(new)
        strata[label] = {"target": m, "generated": len(new)}
    return edges, strata


def build_surrogate(G: nx.Graph, assignment: LayerAssignment, generator: str, rng: np.random.Generator, n_clusters: int = 2) -> tuple[nx.Graph, dict]:
    """
    Assemble one surrogate graph layer by layer (backbone first).

    Returns ``(H, report)``. Every generated edge carries the layer rank it
    was generated at under the `LAYER_KEY` attribute.
    """
    if generator not in GENERATORS:
        raise ValueError(f"Unknown generator {generator}; expected one of {GENERATORS}.")
    if generator == "er_strat":
        node_blocks = _cluster_nodes_by_responsibility(assignment, rng, n_clusters=n_clusters)

    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    taken: set = set()
    report: dict = {"generator": generator, "layers": []}

    for k in range(1, assignment.n_layers + 1):
        original = assignment.layer_edges(k)
        m = len(original)
        layer_nodes = {n for e in original for n in e}
        entry: dict = {"rank": k, "target": m, "generated": 0, "lost": m}

        if m == 0 or len(layer_nodes) < 2:
            if m:
                log.warning("Layer %d: %d edges requested but only %d nodes; skipping.", k, m, len(layer_nodes))
            report["layers"].append(entry)
            continue

        if generator == "er":
            new = _sample_er_layer(sorted(layer_nodes, key=str), m, taken, rng)
        else:
            new, strata = _sample_er_stratified_layer(original, layer_nodes, node_blocks, taken, rng)
            entry["strata"] = strata


        if len(new) < m:
            log.warning("Layer %d (%s): generated %d/%d edges.", k, generator, len(new), m)
        for u, v in new:
            H.add_edge(u, v, **{LAYER_KEY: k})
        entry["generated"] = len(new)
        entry["lost"] = m - len(new)
        report["layers"].append(entry)
    
    log.debug("Surrogate builded successfully")
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
