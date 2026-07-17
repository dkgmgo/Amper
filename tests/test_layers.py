"""Tests for layer extraction and diagnostics."""

import math

import networkx as nx
import numpy as np
import pytest

from layers import LayerAssignment, assign_layers, edge_distances, layer_diagnostics
from mixture import GammaMixture


@pytest.fixture
def toy():
    """
    Two triangles (a,b,c) and (d,e,f) with short edges (distance 1),
    bridged by two long edges (distance 10): c-d and a-e.
    """
    G = nx.Graph()
    intra = [("a", "b"), ("b", "c"), ("c", "a"), ("d", "e"), ("e", "f"), ("f", "d")]
    inter = [("c", "d"), ("a", "e")]
    for u, v in intra:
        G.add_edge(u, v, distance=1.0)
    for u, v in inter:
        G.add_edge(u, v, distance=10.0)
    # Component 0: mean 10 (backbone), component 1: mean 1.
    model = GammaMixture()
    model.weights_= np.array([0.25, 0.75])
    model.alphas_= np.array([40.0, 4.0])
    model.betas_= np.array([4.0, 4.0])
    model.means_ = model.alphas_/model.betas_
    model.converged_ = True
    return G, model, set(map(frozenset, intra)), set(map(frozenset, inter))


def test_backbone_is_largest_mean(toy):
    G, model, intra, inter = toy
    a = assign_layers(G, model)
    assert a.n_layers == 2
    # Rank order must be by decreasing mean.
    assert a.means[0] > a.means[1]
    assert set(map(frozenset, a.layer_edges(1))) == inter
    assert set(map(frozenset, a.layer_edges(2))) == intra


def test_component_order_reindexes_responsibilities(toy):
    G, model, _, _ = toy
    a = assign_layers(G, model)
    np.testing.assert_allclose(a.responsibilities.sum(axis=1), 1.0)
    # Backbone edges must put most responsibility mass on column 0.
    for i, (u, v) in enumerate(a.edges):
        expected_col = 0 if G[u][v]["distance"] > 5 else 1
        assert a.responsibilities[i].argmax() == expected_col


def test_layer_nodes_and_first_layer(toy):
    G, model, _, _ = toy
    a = assign_layers(G, model)
    assert a.layer_nodes(1) == {"a", "c", "d", "e"}  # backbone touches only bridge endpoints
    assert a.layer_nodes(2) == set(G.nodes())  # every node has an intra-triangle edge
    first = a.node_first_layer()
    assert first == {"a": 1, "c": 1, "d": 1, "e": 1, "b": 2, "f": 2}


def test_layer_diagnostics(toy):
    G, model, _, _ = toy
    a = assign_layers(G, model)
    rows = layer_diagnostics(a)
    assert len(rows) == 2
    backbone = rows[0]
    assert backbone["rank"] == 1
    assert backbone["n_edges"] == 2
    assert backbone["n_nodes"] == 4
    assert backbone["n_components"] == 2  # two disjoint bridges
    assert math.isclose(backbone["density"], 2 * 2 / (4 * 3))
    assert rows[1]["n_edges"] == 6
    assert rows[1]["n_components"] == 2  # two triangles


def test_edge_distances_drops_invalid():
    G = nx.Graph()
    G.add_edge("a", "b", distance=1.0)
    G.add_edge("b", "c")  # missing attribute
    G.add_edge("c", "d", distance=float("nan"))
    edges, x = edge_distances(G)
    assert edges == [("a", "b")]
    np.testing.assert_allclose(x, [1.0])


def test_single_component():
    G = nx.path_graph(4)
    nx.set_edge_attributes(G, 1.0, "distance")
    model = GammaMixture()
    model.weights_=np.array([1.0])
    model.alphas_=np.array([4.0])
    model.betas_=np.array([4.0])
    model.means_ = model.alphas_/model.betas_
    a = assign_layers(G, model)
    assert a.n_layers == 1
    assert (a.hard_ranks == 1).all()
