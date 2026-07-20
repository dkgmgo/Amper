"""
Data loading and batch execution utilities.
"""

from __future__ import annotations

import datetime
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import networkx as nx
from tqdm import tqdm
import numpy as np

log = logging.getLogger(__name__)

#: Edge attribute holding the Ricci-flow distance used as filtration value.
DISTANCE_KEY = "distance"

#: Matches e.g. "FR_1702188000.1500.graphml" -> country="FR", ts=1702188000.
_FILENAME_RE = re.compile(r"^(?P<country>[A-Za-z]{2,3})_(?P<ts>\d{9,11})")


@dataclass
class GraphRecord:
    """A loaded graph together with its dataset identity."""

    path: Path
    country: str
    snapshot: Optional[int]
    graph: nx.Graph

    @property
    def snapshot_date(self) -> Optional[str]:
        """Snapshot as an ISO date string (UTC), or `None`."""
        if self.snapshot is None:
            return None
        dt = datetime.datetime.fromtimestamp(self.snapshot, datetime.timezone.utc)
        return dt.date().isoformat()

    @property
    def label(self) -> str:
        """Human-readable identifier, e.g. `FR_2023-12-10`."""
        if self.snapshot_date is not None:
            return f"{self.country}_{self.snapshot_date}"
        return self.country


def parse_country_snapshot(path: Path) -> tuple[str, Optional[int]]:
    """
    Derive (country, snapshot timestamp) from a graph file name.
    """
    m = _FILENAME_RE.match(path.name)
    if m:
        return m.group("country").upper(), int(m.group("ts"))
    log.warning("Could not parse country/snapshot from %s; using stem.", path.name)
    return path.stem, None


def load_graph(path: str | Path, feature_attr: str = DISTANCE_KEY) -> nx.Graph:
    """
    Load a `.graphml` file as an undirected graph with clean distances.

    `feature_attr` names the edge attribute holding the Ricci-flow distance.
    """
    path = Path(path)
    G = nx.read_graphml(path)

    if G.is_multigraph() or G.is_directed():
        H = nx.Graph()
        H.add_nodes_from(G.nodes(data=True))
        edges = (
            G.edges(data=True, keys=False) if G.is_multigraph() else G.edges(data=True)
        )
        for u, v, data in edges:
            if H.has_edge(u, v):
                old = H[u][v].get(feature_attr, math.inf)
                new = data.get(feature_attr, math.inf)
                if new < old:
                    H[u][v].update(data)
            else:
                H.add_edge(u, v, **data)
        G = H

    G.remove_edges_from(nx.selfloop_edges(G))

    # Validate distances.
    valid: list[float] = []
    bad: list[tuple] = []
    for u, v, data in G.edges(data=True):
        d = data.get(feature_attr)
        try:
            d = float(d)
        except (TypeError, ValueError):
            d = math.nan
        if math.isfinite(d) and d >= 0.0:
            valid.append(d)
        else:
            bad.append((u, v))

    if not valid:
        raise ValueError(f"{path}: no edge has a valid '{feature_attr}' attribute; cannot build a distance filtration.")
    if bad:
        log.warning("%s: %d, Bad values found when validating distances", path, len(bad))

    # Descriptive statistics on the features
    global_mean = np.mean(valid)
    global_var = np.var(valid)
    log.info("Global mean: %f", global_mean)
    log.info("Global variance: %f", global_var)

    return G

def write_graphml(G: nx.Graph, path: str | Path) -> None:
    """Write the content of a graph into a .graphml file"""
    nx.write_graphml_lxml(G, path)
    log.info("Graph written successfully")


def load_record(path: str | Path, feature_attr: str = DISTANCE_KEY) -> GraphRecord:
    """Load a graph file together with its parsed dataset identity."""
    path = Path(path)
    country, snapshot = parse_country_snapshot(path)
    return GraphRecord(path=path, country=country, snapshot=snapshot, graph=load_graph(path, feature_attr=feature_attr))


def iter_graph_files(data_dir: str | Path, ignore_dirs: Iterable[str] = ("processed",), pattern: str = "*.graphml") -> list[Path]:
    """
    List graph files under `data_dir` recursively. Any file whose path contains a directory in `ignore_dirs` is skipped.
    """
    data_dir = Path(data_dir)
    ignored = set(ignore_dirs)
    return sorted(
        p for p in data_dir.rglob(pattern)
        if p.is_file() and not ignored.intersection(p.relative_to(data_dir).parts[:-1])
    )


def run_on_all_dataset(fn: Callable[[GraphRecord], object], data_dir: str | Path, ignore_dirs: Iterable[str] = ("processed",), pattern: str = "*.graphml", feature_attr: str = DISTANCE_KEY) -> list[object]:
    """
    Apply `fn` to every graph file under ``data_dir``.
    """
    files = iter_graph_files(data_dir, ignore_dirs=ignore_dirs, pattern=pattern)
    if not files:
        raise FileNotFoundError(f"No '{pattern}' files found under {data_dir}")

    results: list[object] = []
    for path in tqdm(files, desc="Graphs", unit="graph"):
        try:
            record = load_record(path, feature_attr=feature_attr)
            results.append(fn(record))
        except Exception:
            log.exception("Failed on %s; skipping.", path)
    return results
