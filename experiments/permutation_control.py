"""
Permutation control: does the Ricci-flow layer assignment carry information, or do
only the per-layer edge COUNTS matter?

real   surrogates built from the fitted per-edge layer ranks
perm   surrogates built from those same ranks shuffled across the edges, so every
        per-rank edge count is preserved exactly and only *which* edge holds which
        rank is destroyed
"""

from __future__ import annotations

import csv
import logging
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from builder import assign_surrogate_distances, build_surrogate, surrogate_edge_layers
from datasets import discover_graphs, load_record
from layers import LayerAssignment, assign_layers, edge_distances, permute_layer_ranks
from mixture import select_components
from topology import betti_distances, essential_count, get_topology

log = logging.getLogger("perm_control")

GENERATORS = ("ws_hier", "er_hier")
GEN_SEEDS = (0, 1, 2)
PERM_SEEDS: tuple = (None, 101, 102, 103)   # None = real-label
N_RANGE = range(1, 9)
SEED = 0
PERM_SEED_BASE = 1_000_003
EXPANSION_DIM = 2
N_GRID = 512
N_INIT = 5


def _layer_nodes_sizes(assignment: LayerAssignment) -> list[int]:
    return [len(assignment.layer_nodes(k)) for k in range(1, assignment.n_layers + 1)]


def make_perm_assignments(a: LayerAssignment) -> dict:
    """One permuted assignment per seed, shared by the checks and the build."""
    return {
        ps: (a if ps is None
             else permute_layer_ranks(a, np.random.default_rng([PERM_SEED_BASE, int(ps)])))
        for ps in PERM_SEEDS
    }


def fit(graph_path):
    """Load, fit the mixture, assign layers"""
    record = load_record(graph_path)
    _, x = edge_distances(record.graph)
    model, _ = select_components(x, n_range=N_RANGE, seed=SEED, n_init=N_INIT)
    return record, model, assign_layers(record.graph, model)


def print_infos(record, a, perm_assignments):
    G = record.graph
    K = a.n_layers
    real_counts = Counter(int(r) for r in a.hard_ranks)
    real_sizes = _layer_nodes_sizes(a)
    p = np.array([real_counts[k] for k in range(1, K + 1)], dtype=float) / len(a.hard_ranks)

    log.info(f"\n{record.label}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, K={K}")
    log.info(f"n_long = min(2, K//2) = {min(2, K // 2) if K > 1 else 1}")
    log.info(f"real per-rank edge counts: {[real_counts[k] for k in range(1, K + 1)]}")
    log.info(f"real per-rank node counts: {real_sizes}")
    log.info(f"expected relabelling rate 1-sum(p^2): {1 - float((p ** 2).sum()):.1%}")

    log.info(f"{'seed':>6} {'counts identical':>17} {'rank changed':>13}   |layer_nodes(k)|")
    for ps, b in perm_assignments.items():
        if ps is None:
            continue
        # the treatment must preserve every per-rank edge count exactly.
        assert Counter(int(r) for r in b.hard_ranks) == real_counts, \
            f"perm seed {ps}: per-rank edge counts changed"
        # realized relabelling rate, against the 1-sum(p^2) expectation above.
        changed = float(np.mean(np.asarray(a.hard_ranks) != np.asarray(b.hard_ranks)))
        log.info(f"{ps:>6} {'OK':>17} {changed:>12.1%}   {_layer_nodes_sizes(b)}")


def build_arms(record, model, a, perm_assignments) -> tuple[dict, dict, list, np.ndarray, float]:
    """
    build every surrogate before measuring anything
    """
    G = record.graph
    nodes = list(G.nodes())
    original_topology = get_topology(a.edges, a.distances, nodes, EXPANSION_DIM)

    built: dict[str, tuple] = {}
    meta: dict[str, tuple] = {}
    for gi, gen in enumerate(GENERATORS):
        for pi, ps in enumerate(PERM_SEEDS):
            arm = "real" if ps is None else "perm"
            asg = perm_assignments[ps]
            for gs in GEN_SEEDS:
                run = f"{gen}|{arm}|p{ps}|g{gs}"
                rng = np.random.default_rng([SEED, gi, pi, gs])
                dist_rng = np.random.default_rng([SEED, gi, pi, gs, 1])

                H, _ = build_surrogate(G, asg, gen, rng, model.n_components * 2)
                assign_surrogate_distances(H, asg, model, dist_rng)
                edges, _ = surrogate_edge_layers(H)
                _, dists = edge_distances(H)
                built[run] = (H, get_topology(edges, dists, nodes, EXPANSION_DIM), dists)
                meta[run] = (gen, arm, ps, gs)
                log.info(f" Built {run}: {H.number_of_edges()} edges")

    cap = float(max([a.distances.max()] + [d.max() for _, _, d in built.values()]))
    x_range = np.linspace(0.0, cap, N_GRID)
    log.info(f"\nShared filtration scale: cap={cap:.6g} over {N_GRID} grid points.")
    return built, meta, original_topology, x_range, cap


def main(input_path: Path, input_key: str, out_dir: Path) -> None:
    record, model, a = fit(input_path)
    perm_assignments = make_perm_assignments(a)
    print_infos(record, a, perm_assignments)

    G = record.graph
    n_real = len(GEN_SEEDS)
    n_perm = (len(PERM_SEEDS) - 1) * len(GEN_SEEDS)
    log.info(f"RUN: {len(GENERATORS)} generators x ({n_real} real + {n_perm} permuted) seeds "
          f"= {len(GENERATORS) * (n_real + n_perm)} surrogates, one shared cap")
    built, meta, original_topology, x_range, cap = build_arms(record, model, a, perm_assignments)

    orig = {
        "b0": essential_count(original_topology[0]),
        "b1": essential_count(original_topology[1]),
        "C": nx.average_clustering(G),
        "tri": sum(nx.triangles(G).values()) // 3,
        "edges": G.number_of_edges(),
    }

    # measure everything on that shared scale.
    rows: list[dict] = []
    n = G.number_of_nodes()
    for run, (H, surrogate_topology, _) in built.items():
        gen, arm, ps, gs = meta[run]
        # Only betti_l1 is reported, diagram_distances takes to much time for H0 on big graphs.
        diffs = betti_distances(original_topology, surrogate_topology, x_range)
        h0, h1 = float(diffs[(0, "betti_l1")]), float(diffs[(1, "betti_l1")])
        rows.append({
            "dataset": record.label, "K": a.n_layers, "generator": gen, "arm": arm,
            "perm_seed": "" if ps is None else ps, "gen_seed": gs,
            "n_edges_surr": H.number_of_edges(),
            "H0_betti_l1": h0, "H1_betti_l1": h1,
            "beta0": essential_count(surrogate_topology[0]),
            "beta1": essential_count(surrogate_topology[1]),
            "clustering": nx.average_clustering(H),
            "triangles": sum(nx.triangles(H).values()) // 3,
            "H0_per_n": h0 / n, "H1_per_n": h1 / n,
        })

    target = len(a.edges)
    for gen in GENERATORS:
        counts = {r["n_edges_surr"] for r in rows if r["generator"] == gen}
        assert counts == {target}, f"{gen}: expected every run at {target}, got {sorted(counts)}"

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "permutation_control.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log.info(f"\nWrote {csv_path}  ({len(rows)} runs)")

    cols = ["H0_betti_l1", "H1_betti_l1", "H0_per_n", "beta0", "beta1", "clustering", "triangles"]
    log.info(f"SUMMARY {input_key} K={a.n_layers} — mean +/- sd over seeds  (cap={cap:.6g}, shared within this run)")
    log.info(f"original: b0={orig['b0']}  b1={orig['b1']}  C={orig['C']:.4f}  tri={orig['tri']}  edges={orig['edges']}")

    summary_rows = []
    for gen in GENERATORS:
        log.info(f"\n{gen}")
        log.info(f"  {'arm':6s} {'n':>2}  " + "  ".join(f"{c:>19s}" for c in cols))
        for arm in ("real", "perm"):
            sel = [r for r in rows if r["generator"] == gen and r["arm"] == arm]
            if not sel:
                continue
            cells = []
            for c in cols:
                v = np.array([r[c] for r in sel], dtype=float)
                m = float(v.mean())
                s = float(v.std(ddof=1)) if len(v) > 1 else 0.0
                fmt = "{:.5f}" if c in ("clustering", "H0_per_n", "H1_per_n") else "{:.1f}"
                cells.append(f"{fmt.format(m)} +/- {fmt.format(s)}")
                summary_rows.append({"dataset": record.label, "K": a.n_layers, "generator": gen,
                                     "arm": arm, "metric": c, "mean": m, "sd": s, "n": len(v)})
            log.info(f"  {arm:6s} {len(sel):>2}  " + "  ".join(f"{c:>19s}" for c in cells))

    sum_path = out_dir / "permutation_control_summary.csv"
    with open(sum_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "K", "generator", "arm", "metric", "mean", "sd", "n"])
        w.writeheader()
        w.writerows(summary_rows)
    log.info(f"\nWrote {sum_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    graphs = discover_graphs("./data/in", keys=sys.argv[1:] or None)
    log.info("Graphs: %s", ", ".join(f"{k} ({v.name})" for k, v in graphs.items()))

    for key in graphs:
        out_dir = Path(f"./data/out/perm_control_{key}")   
        main(graphs[key], key, out_dir)
