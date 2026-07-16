"""
End-to-end orchestration: load -> mixture -> layers -> surrogates -> topology.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from datasets import DISTANCE_KEY, load_record
from builder import GENERATORS, build_surrogate, assign_surrogate_distances, surrogate_edge_layers
from layers import LayerAssignment, assign_layers, edge_distances, layer_diagnostics
from mixture import GammaMixture, select_components
from topology import get_topology, diagram_distances
from dashboard import plot_dashboard, plot_graphs, plot_model_selection, plot_topology

log = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    input_path: Path
    output_dir: Path
    feature_attr: str = DISTANCE_KEY
    n_range: range = range(1, 8)
    fixed: int = None
    seed: int = 0
    generators: tuple[str, ...] = GENERATORS
    expansion_dim: int = 2
    n_init: int = 5


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
    cfg.n_range = range(cfg.fixed, cfg.fixed + 1) if cfg.fixed else cfg.n_range
    model, bics = select_components(x, n_range=cfg.n_range, seed=cfg.seed, n_init=cfg.n_init)
    plot_model_selection(bics, cfg.n_range, x, model, out / "model_selection.png")

    assignment = assign_layers(G, model, cfg.feature_attr)
    diagnostics = layer_diagnostics(assignment)
    for row in diagnostics:
        log.info(
            "Layer %d: weight=%.3f mean=%.4g edges=%d nodes=%d components=%d density=%.4g",
            row["rank"], row["weight"], row["mean_distance"],
            row["n_edges"], row["n_nodes"], row["n_components"], row["density"],
        )
    _write_csv(out / "layer_diagnostics.csv", list(diagnostics[0].keys()), [list(r.values()) for r in diagnostics])

    original_topology = get_topology(assignment.edges, assignment.distances, cfg.expansion_dim)

    x_range = np.linspace(0, assignment.distances.max() * 1.05, 256)
    #x_range = sorted(assignment.distances)
    rng = np.random.default_rng(cfg.seed)
    dist_rng = np.random.default_rng(cfg.seed)

    metrics: dict[tuple[str, str, int, str], list[float]] = {}
    gen_report_rows: list[list] = []

    for gen in cfg.generators:
        H, report = build_surrogate(G, assignment, gen, rng)
        for entry in report["layers"]:
            gen_report_rows.append([gen, entry["rank"], entry["target"], entry["generated"], entry["lost"]])
        assign_surrogate_distances(H, assignment, model, dist_rng, cfg.feature_attr)
        edges, _ = surrogate_edge_layers(H)
        _, dists = edge_distances(H, cfg.feature_attr)
        surrogate_topology = get_topology(edges, dists, cfg.expansion_dim)
        diffs = diagram_distances(original_topology, surrogate_topology)
        log.info(diffs)
        plot_graphs(G, H, out / "graph.png")
        plot_topology(original_topology, surrogate_topology, out/"topology.png", x_range)
        dashboard_out_path = out / (cfg.input_path.split("/")[-1].split(".graphml")[0] +"_dashboard.png")
        plot_dashboard(bics, cfg.n_range, x, model, G, H, original_topology, surrogate_topology, dashboard_out_path, x_range)

    _write_csv(out / "generation_report.csv", ["generator", "replicate", "rank", "target_edges", "generated_edges", "lost_edges"], gen_report_rows)



    summary_rows: list[list] = []
    _write_csv(out / "summary.csv", ["test"], summary_rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    cfg = PipelineConfig(
        input_path="./data/in/ErdosRenyiRicci.graphml",
        output_dir="./data/out",
        n_range=range(1, 9),
        fixed=1,
        seed=0,
        n_init=5,
    )
    run_pipeline(cfg)