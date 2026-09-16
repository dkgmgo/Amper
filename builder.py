"""
Hierarchical surrogate construction.
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

# Edge attribute recording the layer at which a surrogate edge was generated.
LAYER_KEY = "layer"

# Simple ER generators
_RANDOM = ("er", "er_strat")

# Generators that differ only in how short layers and the long budget are placed.
_LAYERED = {
    #  name        short   long
    "ws_layer":   ("ring", "er"),
    "er_hier":    ("er",   "bridge"),
    "ws_hier":    ("ring", "bridge"),
}

GENERATORS = tuple(list(_RANDOM) + list(_LAYERED.keys()))


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


def _sample_er_stratified_layer(original_edges: Sequence[Edge], layer_nodes: set, node_blocks: Mapping[Hashable, int], taken: set, rng: np.random.Generator) -> list[Edge]:
    """
    Stratified G(n, m): one independent G(n, m) per block pair, each matching the
    original layer's edge count for that pair.
    """
    budgets, nodes_by_block = _block_pair_budgets(original_edges, layer_nodes, node_blocks)

    edges: list[Edge] = []
    for (ba, bb), m in sorted(budgets.items()):
        new = _sample_distinct_pairs(nodes_by_block[ba], nodes_by_block[bb], m, taken, rng, same_set=(ba == bb))
        label = f"{ba}-{bb}"
        if len(new) < m:
            log.warning("Stratum %s: generated %d/%d edges.", label, len(new), m)
        edges.extend(new)
    return edges


def _build_random(G: nx.Graph, assignment: LayerAssignment, generator: str, rng: np.random.Generator, n_clusters: int ):
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
            new = _sample_er_stratified_layer(original, layer_nodes, node_blocks, taken, rng)


        if len(new) < m:
            log.warning("Layer %d (%s): generated %d/%d edges.", k, generator, len(new), m)
        for u, v in new:
            H.add_edge(u, v, **{LAYER_KEY: k})
        entry["generated"] = len(new)
        entry["lost"] = m - len(new)
        report["layers"].append(entry)

    log.debug("Surrogate built successfully")
    return H, report


def _allocate_ranks(count: int, ranks: Sequence[int], sizes: Sequence[int], rng: np.random.Generator) -> list[int]:
    """Split `count` edges across `ranks` in proportion to the original layer `sizes`."""
    if count <= 0 or len(ranks) == 0:
        return []
    w = np.asarray(sizes, dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.full(len(ranks), 1.0 / len(ranks))
    alloc = np.floor(w * count).astype(int)
    while alloc.sum() < count:                     # hand out the rounding remainder
        alloc[int(np.argmax(w * count - alloc))] += 1
    out: list[int] = []
    for r, a in zip(ranks, alloc):
        out.extend([int(r)] * int(a))
    return [out[int(i)] for i in rng.permutation(len(out))]


def _degree_heterogeneous_ring_edges(nodes: Sequence, degrees: np.ndarray, taken: set | None = None) -> list[Edge]:
    """
    Ring lattice with a heterogeneous degree sequence. Node at ring position `i` aims for 
    `degrees[i]` links and takes them from the nearest positions that still have spare capacity
    this mutates taken, and does not guarantee a ring
    """
    n = len(nodes)
    if n < 3:
        return []
    remaining = np.asarray(degrees, dtype=int).copy()
    seen: set[tuple[int, int]] = set()
    out: list[Edge] = []
    taken = taken if taken is not None else set()

    for i in [t for t in np.argsort(-remaining)]: # hubs come first
        j = 1
        while remaining[i] > 0 and j <= n // 2:
            for cand in ((i + j) % n, (i - j) % n):
                if remaining[i] <= 0:
                    break
                if cand == i or remaining[cand] <= 0:
                    continue
                key = (min(i, cand), max(i, cand))
                if key in seen:
                    continue
                ekey = _ekey(nodes[i], nodes[cand])
                if ekey in taken:
                    continue
                seen.add(key)
                taken.add(ekey)
                out.append(ekey)
                remaining[i] -= 1
                remaining[cand] -= 1
            j += 1
    return out


def _hierarchical_pairs(H: nx.Graph, layer_of: Mapping[Hashable, int], m: int, taken: set, rng: np.random.Generator) -> tuple[list[Edge], int]:
    """
    Place `m` links whose endpoints sit in different layers, lower rank first. Connected components 
    of `H` are joined first and the remainder is filled following the layer constraint.
    """
    by_layer: dict[int, list] = defaultdict(list)
    for n, l in layer_of.items():
        by_layer[int(l)].append(n)
    for l in by_layer:
        by_layer[l].sort(key=str)
    layers = sorted(by_layer)

    # union find for connected components
    parent = {n: n for n in H.nodes()}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for u, v in H.edges():
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv

    def allowed_targets(l):
        return [j for j in layers if j > l]

    out: list[Edge] = []
    merged = 0

    # 1) every pair that joins two components and obeys the layer rule
    for l in layers:
        for u in by_layer[l]:
            if len(out) >= m:
                break
            for tl in allowed_targets(l):
                for v in by_layer.get(tl, ()):
                    if len(out) >= m:
                        break
                    if find(u) == find(v):
                        continue
                    key = _ekey(u, v)
                    if key in taken:
                        continue
                    taken.add(key)
                    out.append(key)
                    parent[find(u)] = find(v)
                    merged += 1
                    break

    # 2) fill the rest with any pair obeying the layer rule
    attempts, limit = 0, 60 * m + 2000
    while len(out) < m and attempts < limit:
        attempts += 1
        l = layers[int(rng.integers(len(layers)))]
        tl_choices = allowed_targets(l)
        tl_choices = [t for t in tl_choices if by_layer.get(t)]
        if not tl_choices or not by_layer[l]:
            continue
        tl = tl_choices[int(rng.integers(len(tl_choices)))]
        u = by_layer[l][int(rng.integers(len(by_layer[l])))]
        v = by_layer[tl][int(rng.integers(len(by_layer[tl])))]
        if u == v:
            continue
        key = _ekey(u, v)
        if key in taken:
            continue
        taken.add(key); out.append(key)
    if len(out) < m:
        log.warning("Hierarchical placement: %d/%d links (layer constraint exhausted).", len(out), m)
    return out, merged


def _short_layer_edges(assignment: LayerAssignment, n_long: int, taken: set, rng: np.random.Generator, mode: str) -> dict[int, list[Edge]]:
    """
    Edges for every short layer, each confined to that layer's own node set.
    `mode="ring"` gives a degree-heterogeneous ring. 
    `mode="er"` gives G(n, m).
    """
    out: dict[int, list[Edge]] = {}
    ranks = list(range(n_long + 1, assignment.n_layers + 1))
    for k in ranks:
        original = assignment.layer_edges(k)
        m = len(original)
        layer_nodes = sorted({n for e in original for n in e}, key=str)
        if m == 0 or len(layer_nodes) < 2:
            out[k] = []
            continue
        if mode == "er":
            out[k] = _sample_er_layer(layer_nodes, m, taken, rng)
            continue
        sub = nx.Graph()
        sub.add_nodes_from(layer_nodes)
        sub.add_edges_from(original)
        cn = rng.permutation(layer_nodes)
        deg = np.array([sub.degree(v) for v in cn], dtype=int)
        new = _degree_heterogeneous_ring_edges(cn, deg, taken=taken)
        if len(new) < m:
            new.extend(_sample_distinct_pairs(cn, cn, m - len(new), taken, rng, same_set=True))
        elif len(new) > m:
            keep = set(rng.choice(len(new), size=m, replace=False).tolist())
            taken.difference_update(e for i, e in enumerate(new) if i not in keep)
            log.warning("Layer %d: ring overshot by %d edges; released them from `taken`.", k, len(new) - m)
            new = [new[i] for i in sorted(keep)]
        out[k] = new
    return out


def _build_layered(G: nx.Graph, assignment: LayerAssignment, rng: np.random.Generator, short: str, long: str, name: str) -> tuple[nx.Graph, dict]:
    """
    Build a surrogate layer by layer, short layers first, then the long budget.
    long=bridge merges connected components built with short.
    """
    K = assignment.n_layers
    n_long = min(2, K//2) if K > 1 else 1
    long_sizes = [len(assignment.layer_edges(k)) for k in range(1, n_long + 1)]

    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    taken: set = set()
    entries: list[dict] = []

    for k, edges in sorted(_short_layer_edges(assignment, n_long, taken, rng, short).items()):
        for u, v in edges:
            H.add_edge(u, v, **{LAYER_KEY: k})
        tgt = len(assignment.layer_edges(k))
        entries.append({"rank": k, "target": tgt, "generated": len(edges), "lost": tgt - len(edges)})

    merged = 0
    if long == "bridge":
        placed, merged = _hierarchical_pairs(H, assignment.node_first_layer(), sum(long_sizes), taken, rng)
        tags = _allocate_ranks(len(placed), list(range(1, n_long + 1)), long_sizes, rng) #TODO improve this
        for (u, v), r in zip(placed, tags):
            H.add_edge(u, v, **{LAYER_KEY: r})
    else:
        for k in range(1, n_long + 1):
            original = assignment.layer_edges(k)
            ln = sorted({n for e in original for n in e}, key=str)
            if original and len(ln) >= 2:
                for u, v in _sample_er_layer(ln, len(original), taken, rng):
                    H.add_edge(u, v, **{LAYER_KEY: k})
    for k in range(1, n_long + 1):
        tgt = len(assignment.layer_edges(k))
        got = sum(1 for _, _, d in H.edges(data=True) if int(d[LAYER_KEY]) == k)
        entries.append({"rank": k, "target": tgt, "generated": got, "lost": tgt - got})
    entries.sort(key=lambda e: e["rank"])

    m_total = len(assignment.edges)
    p = sum(long_sizes) / m_total if m_total else 0.0
    log.info("%s: edges=%d (target %d) p=%.4f n_long=%d merged_components=%d", name, H.number_of_edges(), m_total, p, n_long, merged)
    return H, {"generator": name, "layers": entries}


def build_surrogate(G: nx.Graph, assignment: LayerAssignment, generator: str, rng: np.random.Generator, n_clusters: int = 2) -> tuple[nx.Graph, dict]:
    """
    Assemble one surrogate graph

    Returns ``(H, report)``. Every generated edge carries the layer rank it
    was generated at under the `LAYER_KEY` attribute.
    """
    if generator not in GENERATORS:
        raise ValueError(f"Unknown generator {generator}; expected one of {GENERATORS}.")
    if generator in _RANDOM:
        return _build_random(G, assignment, generator, rng, n_clusters)
    if generator in _LAYERED:
        short, long = _LAYERED[generator]
        return _build_layered(G, assignment, rng, short, long, generator)


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
