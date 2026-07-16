"""
Plotting utilities.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

import networkx as nx
import numpy as np
from scipy.stats import gamma

from mixture import GammaMixture

log = logging.getLogger(__name__)

Diagram = np.ndarray
Diagrams = list[Diagram]


def _pyplot():
    """Lazy import so matplotlib is only required when plotting."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _draw_bic_curve(ax, bics: Sequence[float], n_range: range) -> None:
    """Draw the BIC-vs-K curve on a given axis, marking the selected K."""
    ks = list(n_range)
    ax.plot(ks, bics, "o-", label="BIC", linewidth=2, color="blue")
    best = ks[int(np.argmin(bics))]
    ax.axvline(best, color="red", linestyle="--", label=f"selected K={best}")
    ax.set_xlabel("number of components K")
    ax.set_ylabel("BIC")
    ax.set_title("Gamma mixture model selection")
    ax.legend()
    ax.grid(True)


def _draw_mixture_fit(ax, X: Iterable[float], model: GammaMixture, bins: int = 60) -> None:
    """Draw the fitted mixture density over the empirical histogram on a given axis."""
    plt = _pyplot()
    X = np.array(X)

    # histogram
    ax.hist(X, bins=bins, density=True, alpha=0.8, color="lightgray", edgecolor="white", label="Edge Distances")

    x = np.linspace(X.min(), X.max(), 512)
    pdfs = []
    for k in range(model.n_components):
        alpha = model.alphas_[k]
        beta = model.betas_[k]
        pdfs.append(gamma.pdf(x, a=alpha, scale=1 / beta))

    pdfs = np.vstack(pdfs).T
    total = np.zeros_like(x)

    # plot components
    cmap = plt.get_cmap("tab10")
    order = np.argsort(model.means_.ravel())
    for rank, k in enumerate(order):
        weight = model.weights_[k]
        y = weight * pdfs[:, k]
        ax.plot(x, y, label=f"Cluster {rank} (mean={model.means_[k]:.3g}, w={model.weights_[k]:.2f})", linewidth=2, color=cmap((rank - 1) % 10))
        total += y

    # total mixture
    ax.plot(x, total, "k--", label="Total Mixture", linewidth=1.5)

    ax.set_xlabel("Edge Distance")
    ax.set_ylabel("Density")
    ax.set_title(f"Gamma mixture fit (K={model.n_components})")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _draw_graph(ax, G: nx.Graph, title: str, seed: int = 42, node_color: str = "steelblue") -> None:
    """Draw a networkx graph on a given axis."""
    pos = nx.spring_layout(G, seed=seed)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4, width=0.8)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=60, node_color=node_color, linewidths=0.5, edgecolors="white")
    ax.set_title(f"{title}\n({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    ax.set_axis_off()


def _to_gudhi_persistence(diagrams: Diagrams, dims: Sequence[int]) -> list:
    """Convert per-dimension diagram arrays to gudhi's [(dim, (birth, death)), ...] format."""
    pers = []
    for dim in dims:
        if dim >= len(diagrams) or diagrams[dim] is None or len(diagrams[dim]) == 0:
            continue
        for birth, death in np.asarray(diagrams[dim], dtype=float):
            pers.append((int(dim), (float(birth), float(death))))
    return pers
 
 
def _draw_persistence_diagram(ax, diagrams: Diagrams, title: str, dims: Sequence[int] = (0, 1)) -> None:
    """Draw persistence diagrams on a given axis using gudhi's internal plotter."""
    import gudhi as gh
    pers = _to_gudhi_persistence(diagrams, dims)
    gh.plot_persistence_diagram(persistence=pers, axes=ax, legend=True, fontsize=10)
    ax.set_title(title)
 
 
def _betti_curve(diagram: Diagram, x_range: np.ndarray) -> np.ndarray:
    """
    Betti curve: number of intervals alive at each x_range value.
    Works directly with +inf deaths (inf > R is always True), so
    essential features stay alive over the whole x_range — no cap needed.
    """
    if diagram is None or len(diagram) == 0:
        return np.zeros_like(x_range, dtype=int)
    d = np.asarray(diagram, dtype=float)
    births = d[:, 0]
    deaths = d[:, 1]
    return ((births[None, :] <= x_range[:, None]) & (deaths[None, :] > x_range[:, None])).sum(axis=1)
 
 
def _draw_betti_curves(ax, diagrams: Diagrams, title: str, x_range: np.ndarray, dims: Sequence[int] = (0, 1)) -> None:
    """Draw Betti curves (all homology dimensions) on a given axis."""
    plt = _pyplot()
    cmap = plt.get_cmap("tab10")
 
    for dim in dims:
        if dim >= len(diagrams):
            continue
        betti = _betti_curve(diagrams[dim], x_range)
        ax.step(x_range, betti, where="post", linewidth=2, color=cmap(dim), label=f"$\\beta_{dim}$")
 
    ax.set_xlabel("filtration value")
    ax.set_ylabel("Betti number")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_model_selection(bics: Sequence[float], n_range: range, X: Iterable[float], model: GammaMixture, path: str | Path, bins: int = 60) -> None:
    """Save the BIC curve and the mixture fit side by side in a single figure."""
    plt = _pyplot()
    fig, (ax_bic, ax_fit) = plt.subplots(1, 2, figsize=(13, 4.5))
    _draw_bic_curve(ax_bic, bics, n_range)
    _draw_mixture_fit(ax_fit, X, model, bins=bins)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Model selection plot saved to {path}")


def plot_graphs(G_original: nx.Graph, G_surrogate: nx.Graph, path: str | Path, seed: int = 42) -> None:
    """Save the original and surrogate graphs side by side in a single figure."""
    plt = _pyplot()
    fig, (ax_o, ax_s) = plt.subplots(1, 2, figsize=(13, 5.5))
    _draw_graph(ax_o, G_original, "Original graph", seed=seed, node_color="steelblue")
    _draw_graph(ax_s, G_surrogate, "Surrogate graph", seed=seed, node_color="darkorange")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Graph comparison plot saved to {path}")


def plot_topology(original: Diagrams, surrogate: Diagrams, path: str | Path, x_range: np.ndarray, dims: Sequence[int] = (0, 1)) -> None:
    """Save topology summary in a single figure (2x2)"""
    plt = _pyplot()
 
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    _draw_persistence_diagram(axes[0, 0], original, "Persistence diagram - original", dims=dims)
    _draw_persistence_diagram(axes[0, 1], surrogate, "Persistence diagram - surrogate", dims=dims)
    _draw_betti_curves(axes[1, 0], original, "Betti curves - original", x_range=x_range, dims=dims)
    _draw_betti_curves(axes[1, 1], surrogate, "Betti curves - surrogate", x_range=x_range, dims=dims)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Topology plot saved to {path}")



def plot_dashboard(bics: Sequence[float], n_range: range, X: Iterable[float], model: GammaMixture, G_original: nx.Graph, G_surrogate: nx.Graph,
    original: Diagrams, surrogate: Diagrams, path: str | Path, x_range: np.ndarray, dims: Sequence[int] = (0, 1), bins: int = 60, seed: int = 42) -> None:
    """Save the full dashboard to PNG on a 3x2 gridspec"""
    plt = _pyplot()
 
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)
 
    # row 1: model selection
    _draw_bic_curve(fig.add_subplot(gs[0, 0]), bics, n_range)
    _draw_mixture_fit(fig.add_subplot(gs[0, 1]), X, model, bins=bins)
 
    # row 2: graphs
    _draw_graph(fig.add_subplot(gs[1, 0]), G_original, "Original graph", seed=seed, node_color="steelblue")
    _draw_graph(fig.add_subplot(gs[1, 1]), G_surrogate, "Surrogate graph", seed=seed, node_color="darkorange")
 
    # row 3: topology
    gs_pers = gs[2, 0].subgridspec(1, 2, wspace=0.35)
    _draw_persistence_diagram(fig.add_subplot(gs_pers[0, 0]), original, "Persistence - original", dims=dims)
    _draw_persistence_diagram(fig.add_subplot(gs_pers[0, 1]), surrogate, "Persistence - surrogate", dims=dims)
 
    gs_bc = gs[2, 1].subgridspec(1, 2, wspace=0.35)
    _draw_betti_curves(fig.add_subplot(gs_bc[0, 0]), original, "Betti - original", x_range=x_range, dims=dims)
    _draw_betti_curves(fig.add_subplot(gs_bc[0, 1]), surrogate, "Betti - surrogate", x_range=x_range, dims=dims)
 
    fig.suptitle("Analysis dashboard", fontsize=16, y=0.995)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Dashboard saved to {path}")
