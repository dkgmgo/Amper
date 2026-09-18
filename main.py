"""
End-to-end orchestration: load -> mixture -> layers -> surrogates -> topology.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import networkx as nx

from datasets import DISTANCE_KEY, load_record, write_graphml
from builder import GENERATORS, build_surrogate, assign_surrogate_distances, surrogate_edge_layers
from layers import assign_layers, edge_distances, layer_diagnostics
from mixture import select_components
from topology import get_topology, diagram_distances, betti_distances, essential_count
from dashboard import plot_dashboard, plot_graphs, plot_model_selection, plot_topology

log = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    input_path: str | Path
    output_dir: str | Path
    feature_attr: str = DISTANCE_KEY
    n_range: range = range(1, 8)
    fixed: int | None = None
    seed: int = 0
    generators: tuple[str, ...] = GENERATORS
    expansion_dim: int = 2
    n_init: int = 5
    #: Wasserstein variant for H1 and above; H0 is always exact.
    wasserstein_method: str = "sliced"
    #: Projection count for the sliced variant
    num_directions: int = 50
    #: Grid resolution for Betti-curve comparison and plotting.
    n_grid: int = 512
    #: Generation seeds.
    gen_seeds: tuple[int, ...] = (0,)
    plot_seed_index: int = 0


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    log.info("Wrote %s", path)


def run_pipeline(cfg: PipelineConfig) -> dict:
    """Run the full pipeline"""
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    record = load_record(cfg.input_path, feature_attr=cfg.feature_attr)
    G = record.graph
    log.info("Loaded %s: %d nodes, %d edges.", record.label, G.number_of_nodes(), G.number_of_edges())

    _, x = edge_distances(G, cfg.feature_attr)
    n_range = range(cfg.fixed, cfg.fixed + 1) if cfg.fixed else cfg.n_range
    model, bics = select_components(x, n_range=n_range, seed=cfg.seed, n_init=cfg.n_init)
    plot_model_selection(bics, n_range, x, model, out / "model_selection.png")

    assignment = assign_layers(G, model, cfg.feature_attr)
    diagnostics = layer_diagnostics(assignment)
    for row in diagnostics:
        log.info(
            "Layer %d: weight=%.3f mean=%.4g edges=%d nodes=%d components=%d density=%.4g",
            row["rank"], row["weight"], row["mean_distance"],
            row["n_edges"], row["n_nodes"], row["n_components"], row["density"],
        )
    _write_csv(out / "layer_diagnostics.csv", list(diagnostics[0].keys()), [list(r.values()) for r in diagnostics])

    nodes = list(G.nodes())
    original_topology = get_topology(assignment.edges, assignment.distances, nodes, cfg.expansion_dim)
    orig_triangles = sum(nx.triangles(G).values()) // 3

    metrics: dict[tuple[str, str, int, int, str], float] = {}
    gen_report_rows: list[list] = []
    structure_rows: list[list] = []

    # Pass 1: build every surrogate first, over every seed, so one cap covers them all.
    built: dict[tuple[str, int], tuple] = {}
    for gi, gen in enumerate(cfg.generators):
        for si, sd in enumerate(cfg.gen_seeds):
            rng = np.random.default_rng([cfg.seed, gi, sd])
            dist_rng = np.random.default_rng([cfg.seed, gi, sd, 1])

            H, report = build_surrogate(G, assignment, gen, rng, model.n_components*2)
            for entry in report["layers"]:
                gen_report_rows.append([gen, sd, entry["rank"], entry["target"], entry["generated"], entry["lost"]])
            assign_surrogate_distances(H, assignment, model, dist_rng, cfg.feature_attr)
            edges, _ = surrogate_edge_layers(H)
            _, dists = edge_distances(H, cfg.feature_attr)
            built[(gen, sd)] = (H, get_topology(edges, dists, nodes, cfg.expansion_dim), dists)

    cap = float(max([assignment.distances.max()] + [d.max() for _, _, d in built.values()]))
    x_range = np.linspace(0.0, cap, cfg.n_grid)
    log.info("Shared filtration scale: cap=%.6g over %d grid points.", cap, cfg.n_grid)

    # Pass 2: measure and plot everything on that shared scale.
    plot_seed = cfg.gen_seeds[cfg.plot_seed_index] if cfg.gen_seeds else None
    for (gen, sd), (H, surrogate_topology, _) in built.items():
        diffs = diagram_distances(original_topology, surrogate_topology, cap=cap, wasserstein_method=cfg.wasserstein_method, num_directions=cfg.num_directions)
        diffs.update(betti_distances(original_topology, surrogate_topology, x_range))
        for (dim, metric), value in diffs.items():
            metrics[(record.label, gen, sd, dim, metric)] = float(value)
        log.info("%s seed=%d: %s", gen, sd, {f"H{d}-{m}": round(v, 5) for (d, m), v in sorted(diffs.items())})

        # Structural counts kept out of the distances
        surr_triangles = sum(nx.triangles(H).values()) // 3
        log.info("Triangles orig vs surr: %d vs %d", orig_triangles, surr_triangles)
        structure_rows.append([
            record.label, gen, sd, G.number_of_nodes(), G.number_of_edges(), H.number_of_edges(),
            essential_count(original_topology[0]), essential_count(surrogate_topology[0]),
            essential_count(original_topology[1]), essential_count(surrogate_topology[1]),
            orig_triangles, surr_triangles,
        ])

        if sd != plot_seed:
            continue
        stem = f"{record.label}_{gen}"
        plot_graphs(G, H, out / f"{gen}_graph.png", model=model)
        plot_topology(original_topology, surrogate_topology, out / f"{gen}_topology.png", x_range)
        plot_dashboard(bics, n_range, x, model, G, H, original_topology, surrogate_topology, out / f"{stem}_dashboard.png", x_range)
        #write_graphml(H, out/f"{stem}_surrogate.graphml")

    _write_csv(out / "structure.csv",
               ["dataset", "generator", "seed", "n_nodes", "n_edges_orig", "n_edges_surr",
                "b0_orig", "b0_surr", "b1_orig", "b1_surr", "triangles_orig", "triangles_surr"],
               structure_rows)

    _write_csv(out / "generation_report.csv", ["generator", "seed", "rank", "target_edges", "generated_edges", "lost_edges"], gen_report_rows)

    summary_rows = [[label, gen, sd, dim, metric, value] for (label, gen, sd, dim, metric), value in sorted(metrics.items(), key=lambda kv: kv[0][1:])]
    _write_csv(out / "summary.csv", ["dataset", "generator", "seed", "dim", "metric", "distance"], summary_rows)

    # Mean/sd per (generator, dim, metric) over the seeds
    agg: dict[tuple, list[float]] = {}
    for (label, gen, sd, dim, metric), value in metrics.items():
        agg.setdefault((label, gen, dim, metric), []).append(value)
    summary_agg_rows = [
        [label, gen, dim, metric, len(v), float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0]
        for (label, gen, dim, metric), v in sorted(agg.items(), key=lambda kv: kv[0][1:])
    ]
    _write_csv(out / "summary_agg.csv",
               ["dataset", "generator", "dim", "metric", "n_seeds", "mean", "sd"], summary_agg_rows)

    return {
        "record": record,
        "model": model,
        "bics": bics,
        "assignment": assignment,
        "diagnostics": diagnostics,
        "original_topology": original_topology,
        "metrics": metrics,
        "built": built,
        "cap": cap,
        "x_range": x_range,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    cfg = PipelineConfig(
        input_path="./data/in/CN_1702188000.3000.graphml",
        output_dir="./data/out",
        n_range=range(1, 9),
        fixed=None,
        seed=0,
        n_init=5,
    )
    run_pipeline(cfg)