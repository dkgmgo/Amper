"""
Generator results table.
"""

from __future__ import annotations

import csv
import logging
import statistics as st
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main as pipeline
from datasets import discover_graphs
from builder import GENERATORS
from main import PipelineConfig, run_pipeline

log = logging.getLogger("gen_results")

DATA_DIR = "./data/in"
GRAPHS = discover_graphs(DATA_DIR, keys=sys.argv[1:] or None)
SEEDS = (0, 1, 2)
SEED = 0 # drives the mixture fit (once per graph)
N_RANGE = range(1, 9)
FIXED_K: int | None = None
OUT_ROOT = Path("./data/out/gen_results")
SKIP_PLOTS = True
SKIP_DIAGRAM_DISTANCES = True


class _LogCapture(logging.Handler):
    """Pull `cap` and per-generator `merged_components` out of the pipeline's log records."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.cap: float | None = None
        self.merged: dict[str, int] = {}

    def reset(self) -> None:
        self.cap = None
        self.merged = {}

    def emit(self, record: logging.LogRecord) -> None:
        msg, args = str(record.msg), record.args
        if not args:
            return
        if "Shared filtration scale" in msg:
            self.cap = float(args[0])
        elif "merged_components" in msg:
            self.merged[str(args[0])] = int(args[-1])


def _install(capture: _LogCapture) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    root = logging.getLogger()
    for h in root.handlers:
        h.setLevel(logging.WARNING)
    root.setLevel(logging.INFO)
    root.addHandler(capture)

    if SKIP_PLOTS:
        noop = lambda *a, **k: None
        for fn in ("plot_graphs", "plot_topology", "plot_dashboard", "plot_model_selection"):
            setattr(pipeline, fn, noop)
    if SKIP_DIAGRAM_DISTANCES:
        setattr(pipeline, "diagram_distances", lambda *a, **k: {})


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def run_graph(key: str, path: str, capture: _LogCapture) -> list[dict]:
    """One `run_pipeline` invocation covering every seed; one row per (generator, seed)."""
    out = OUT_ROOT / key
    capture.reset()
    cfg = PipelineConfig(input_path=path, output_dir=out, n_range=N_RANGE, fixed=FIXED_K,
                         seed=SEED, gen_seeds=SEEDS, generators=GENERATORS)
    res = run_pipeline(cfg)
    G, label = res["record"].graph, res["record"].label
    cap, n, K = res["cap"], G.number_of_nodes(), res["assignment"].n_layers

    structure = {(r["generator"], int(r["seed"])): r for r in _read_csv(out / "structure.csv")}
    betti = {(r["generator"], int(r["seed"]), r["dim"]): float(r["distance"])
             for r in _read_csv(out / "summary.csv") if r["metric"] == "betti_l1"}
    lost: dict[tuple[str, int], int] = {}
    for r in _read_csv(out / "generation_report.csv"):
        k = (r["generator"], int(r["seed"]))
        lost[k] = lost.get(k, 0) + int(r["lost_edges"])

    orig_C = nx.average_clustering(G)
    rows = []
    for (gen, sd), (H, _topo, _d) in res["built"].items():
        st_ = structure[(gen, sd)]
        h0, h1 = betti[(gen, sd, "0")], betti[(gen, sd, "1")]
        rows.append({
            "dataset": label, "key": key, "seed": sd, "K": K, "generator": gen,
            "n_nodes": n, "n_edges_orig": int(st_["n_edges_orig"]),
            "n_edges_surr": int(st_["n_edges_surr"]), "lost_edges": lost.get((gen, sd), 0),
            "merged": capture.merged.get(gen, ""),
            "C_orig": orig_C, "C_surr": nx.average_clustering(H),
            "b0_orig": int(st_["b0_orig"]), "b0_surr": int(st_["b0_surr"]),
            "b1_orig": int(st_["b1_orig"]), "b1_surr": int(st_["b1_surr"]),
            "tri_orig": int(st_["triangles_orig"]), "tri_surr": int(st_["triangles_surr"]),
            "H0_betti_l1": h0, "H1_betti_l1": h1,
            "H0_per_n": h0 / n, "H1_per_n": h1 / n,
            "cap": cap,
        })
    return rows


AGG = ["C_surr", "b0_surr", "b1_surr", "tri_surr", "H0_betti_l1", "H1_betti_l1",
       "H0_per_n", "lost_edges", "cap"]


def summarise(key: str, rows: list[dict]) -> list[dict]:
    """Mean/sd per generator over seeds, and print the table."""
    first = rows[0]
    ks = sorted({r["K"] for r in rows})
    print(f"{first['dataset']}  n={first['n_nodes']}  m={first['n_edges_orig']}  "
          f"seeds={sorted({r['seed'] for r in rows})}  K={ks[0] if len(ks) == 1 else ks}")
    if len(ks) > 1:
        log.warning("K differs across seeds, some seeds picked a different K.")
    print(f"original:  C={first['C_orig']:.4f}  b0={first['b0_orig']}  "
          f"b1={first['b1_orig']}  triangles={first['tri_orig']}")
    print(f"cap={first['cap']:.4f}, shared across every seed and generator "
          f"(betti_l1 is on this scale; cap-invariant where beta0 matches the original)")
    print(f"{'generator':10} {'C':>16} {'b0':>9} {'b1':>13} {'triangles':>17} "
          f"{'H0_betti_l1':>15} {'H0_per_n':>9} {'lost':>5} {'merged':>7}")

    out_rows = []
    for gen in GENERATORS:
        sel = [r for r in rows if r["generator"] == gen]
        if not sel:
            continue
        m = lambda c: st.mean([float(r[c]) for r in sel])
        sd = lambda c: st.stdev([float(r[c]) for r in sel]) if len(sel) > 1 else 0.0
        merged = [r["merged"] for r in sel if r["merged"] != ""]
        print(f"{gen:10} {m('C_surr'):>8.4f}±{sd('C_surr'):<7.4f} "
              f"{m('b0_surr'):>4.1f}±{sd('b0_surr'):<4.1f} "
              f"{m('b1_surr'):>7.1f}±{sd('b1_surr'):<5.1f} "
              f"{m('tri_surr'):>10.0f}±{sd('tri_surr'):<6.0f} "
              f"{m('H0_betti_l1'):>8.1f}±{sd('H0_betti_l1'):<6.1f} "
              f"{m('H0_per_n'):>9.5f} {m('lost_edges'):>5.0f} "
              f"{(st.mean(map(int, merged)) if merged else float('nan')):>7.1f}")
        row = {"dataset": first["dataset"], "key": key, "generator": gen,
               "n_seeds": len(sel), "K": ",".join(str(k) for k in ks),
               "n_nodes": first["n_nodes"],
               "n_edges_orig": first["n_edges_orig"], "C_orig": first["C_orig"],
               "b0_orig": first["b0_orig"], "b1_orig": first["b1_orig"],
               "tri_orig": first["tri_orig"],
               "merged_mean": st.mean(map(int, merged)) if merged else ""}
        for c in AGG:
            row[f"{c}_mean"], row[f"{c}_sd"] = m(c), sd(c)
        out_rows.append(row)
    return out_rows


def main() -> None:
    capture = _LogCapture()
    _install(capture)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_raw, all_agg = [], []
    for key, path in GRAPHS.items():
        print(f"\n### {key} — {len(SEEDS)} seeds x {len(GENERATORS)} generators, one shared cap", flush=True)
        rows = run_graph(key, path, capture)
        d = OUT_ROOT / key
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "gen_results.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"Wrote {d / 'gen_results.csv'} ({len(rows)} rows)")
        agg = summarise(key, rows)
        with open(d / "gen_results_summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(agg[0].keys())); w.writeheader(); w.writerows(agg)
        all_raw.extend(rows); all_agg.extend(agg)

    with open(OUT_ROOT / "gen_results_all.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_raw[0].keys())); w.writeheader(); w.writerows(all_raw)
    with open(OUT_ROOT / "gen_results_summary_all.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_agg[0].keys())); w.writeheader(); w.writerows(all_agg)
    print(f"\nWrote {OUT_ROOT / 'gen_results_all.csv'} and {OUT_ROOT / 'gen_results_summary_all.csv'}")


if __name__ == "__main__":
    main()
