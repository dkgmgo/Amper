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
from topology import get_topology, diagram_distances
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
    cap = None

    metrics: dict[tuple[str, str, int, str], float] = {}
    gen_report_rows: list[list] = []

    for gi, gen in enumerate(cfg.generators):
        rng = np.random.default_rng([cfg.seed, gi])
        dist_rng = np.random.default_rng([cfg.seed, gi, 1])

        H, report = build_surrogate(G, assignment, gen, rng, model.n_components*2)
        for entry in report["layers"]:
            gen_report_rows.append([gen, entry["rank"], entry["target"], entry["generated"], entry["lost"]])
        assign_surrogate_distances(H, assignment, model, dist_rng, cfg.feature_attr)
        edges, _ = surrogate_edge_layers(H)
        log.info("Number of triangles orig vs surr: %d vs %d", sum(nx.triangles(G).values()) // 3, sum(nx.triangles(H).values()) // 3)
        _, dists = edge_distances(H, cfg.feature_attr)
        surrogate_topology = get_topology(edges, dists, nodes, cfg.expansion_dim)

        cap = max(assignment.distances.max(), dists.max())
        diffs = diagram_distances(original_topology, surrogate_topology, cap=cap, wasserstein_method=cfg.wasserstein_method, num_directions=cfg.num_directions)
        for (dim, metric), value in diffs.items():
            metrics[(record.label, gen, dim, metric)] = float(value)
        log.info("%s: %s", gen, {f"H{d}-{m}": round(v, 5) for (d, m), v in sorted(diffs.items())})

        x_range = np.linspace(0, cap * 1.05, 256)
        #x_range = sorted(assignment.distances)
        stem = f"{record.label}_{gen}"
        plot_graphs(G, H, out / f"{gen}_graph.png", model=model)
        plot_topology(original_topology, surrogate_topology, out / f"{gen}_topology.png", x_range)
        plot_dashboard(bics, n_range, x, model, G, H, original_topology, surrogate_topology, out / f"{stem}_dashboard.png", x_range)
        write_graphml(G, out/f"{stem}_surrogate.graphml")

    _write_csv(out / "generation_report.csv", ["generator", "rank", "target_edges", "generated_edges", "lost_edges"], gen_report_rows)

    summary_rows = [[label, gen, dim, metric, value] for (label, gen, dim, metric), value in sorted(metrics.items(), key=lambda kv: kv[0][1:])]
    _write_csv(out / "summary.csv", ["dataset", "generator", "dim", "metric", "distance"], summary_rows)

    return {
        "record": record,
        "model": model,
        "bics": bics,
        "assignment": assignment,
        "diagnostics": diagnostics,
        "original_topology": original_topology,
        "metrics": metrics,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    cfg = PipelineConfig(
        input_path="./data/in/ErdosRenyiRicci.graphml",
        output_dir="./data/out",
        n_range=range(1, 9),
        fixed=3,
        seed=0,
        n_init=5,
    )
    run_pipeline(cfg)