"""Tests for hierarchical surrogate generators."""

import networkx as nx
import numpy as np
import pytest

from builder import GENERATORS, LAYER_KEY, DISTANCE_KEY, assign_surrogate_distances, build_surrogate, surrogate_edge_layers
from layers import assign_layers
from mixture import GammaMixture


@pytest.fixture
def toy():
    """Two triangles bridged by two long edges (see test_layers)."""
    G = nx.Graph()
    for u, v in [("a", "b"), ("b", "c"), ("c", "a"), ("d", "e"), ("e", "f"), ("f", "d")]:
        G.add_edge(u, v, distance=1.0)
    for u, v in [("c", "d"), ("a", "e")]:
        G.add_edge(u, v, distance=10.0)
    model = GammaMixture()
    model.weights_= np.array([0.25, 0.75])
    model.alphas_= np.array([40.0, 4.0])
    model.betas_= np.array([4.0, 4.0])
    model.means_ = model.alphas_/model.betas_
    model.converged_ = True
    return G, model, assign_layers(G, model)


@pytest.mark.parametrize("generator", GENERATORS)
def test_node_constraint_and_layer_tags(toy, generator):
    G, _, a = toy
    rng = np.random.default_rng(0)
    H, _ = build_surrogate(G, a, generator, rng)
    assert set(H.nodes()) == set(G.nodes())  # all nodes present from the start
    for u, v, data in H.edges(data=True):
        k = data[LAYER_KEY]
        allowed = a.layer_nodes(k)
        assert u in allowed and v in allowed


@pytest.mark.parametrize("generator", ["er"])
def test_edge_counts_match(toy, generator):
    G, _, a = toy
    H, report = build_surrogate(G, a, generator, np.random.default_rng(1))
    for entry in report["layers"]:
        assert entry["generated"] == entry["target"]
        assert entry["lost"] == 0
    assert H.number_of_edges() == G.number_of_edges()


def test_surrogate_distances_sampled_from_layer_component(toy):
    G, model, a = toy
    H, _ = build_surrogate(G, a, "er", np.random.default_rng(4))
    assign_surrogate_distances(H, a, model, np.random.default_rng(5))
    edges, ranks = surrogate_edge_layers(H)
    for (u, v), k in zip(edges, ranks):
        d = H[u][v][DISTANCE_KEY]
        # Component means are 10 (rank 1) and 1 (rank 2);
        if k == 1:
            assert d > 4.0
        else:
            assert d < 4.0
