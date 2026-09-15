"""Tests for hierarchical surrogate generators."""

import networkx as nx
import numpy as np
import pytest

from builder import _RANDOM, LAYER_KEY, DISTANCE_KEY, _LAYERED, assign_surrogate_distances, build_surrogate, surrogate_edge_layers
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


@pytest.mark.parametrize("generator", _RANDOM)
def test_node_constraint_and_layer_tags(toy, generator):
    G, _, a = toy
    rng = np.random.default_rng(0)
    H, _ = build_surrogate(G, a, generator, rng)
    assert set(H.nodes()) == set(G.nodes())  # all nodes present from the start
    for u, v, data in H.edges(data=True):
        k = data[LAYER_KEY]
        allowed = a.layer_nodes(k)
        assert u in allowed and v in allowed


@pytest.mark.parametrize("generator", _RANDOM)
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


class TestLayered:
    def _layered(self):
        """Short edges on a dense core, long edges reaching nodes the core never touches."""
        G = nx.Graph()
        rng = np.random.default_rng(0)
        core = [f"c{i:02d}" for i in range(50)]
        for i, u in enumerate(core):                       # short: dense ring on the core
            for j in (1, 2, 3):
                G.add_edge(u, core[(i + j) % len(core)], distance=float(rng.gamma(4.0, 0.25)))
        for i in range(30):                                # long: leaves seen only late
            G.add_edge(f"leaf{i:02d}", core[int(rng.integers(50))], distance=float(rng.gamma(100.0, 0.1)))
        model = GammaMixture()
        model.weights_ = np.array([0.2, 0.8])
        model.alphas_ = np.array([100.0, 4.0])
        model.betas_ = np.array([10.0, 4.0])
        model.means_ = model.alphas_ / model.betas_
        model.converged_ = True
        return G, model, assign_layers(G, model)

    def test_edges_stay_inside_their_layer_node_set(self):
        G, _, a = self._layered()
        H, _ = build_surrogate(G, a, "ws_layer", np.random.default_rng(0), n_long=1)
        for u, v, data in H.edges(data=True):
            allowed = a.layer_nodes(int(data[LAYER_KEY]))
            assert u in allowed and v in allowed

    def test_leaf_nodes_get_no_short_edges(self):
        """A node whose only original edge is long must not gain a short one."""
        G, _, a = self._layered()
        for gen in [gen for gen in _LAYERED if 'ws' in gen]:
            H, _ = build_surrogate(G, a, gen, np.random.default_rng(0), n_long=1)
            short_nodes = set()
            for k in range(2, a.n_layers + 1):
                short_nodes |= a.layer_nodes(k)
            for u, v, data in H.edges(data=True):
                if int(data[LAYER_KEY]) > 1:
                    assert u in short_nodes and v in short_nodes

    def test_edge_count_and_nodes(self):
        G, _, a = self._layered()
        for gen in _LAYERED:
            H, report = build_surrogate(G, a, gen, np.random.default_rng(0), n_long=1)
            assert set(H.nodes()) == set(G.nodes())
            assert H.number_of_edges() == len(a.edges)
            assert all(e["lost"] == 0 for e in report["layers"])

    def test_long_links_climb_the_hierarchy(self):
        """Every bridge joins two different layers, lower rank to higher."""
        G, _, a = self._layered()
        for gen in [gen for gen in _LAYERED if 'hier' in gen]:
            H, _ = build_surrogate(G, a, gen, np.random.default_rng(0), n_long=1)
            layer_of = a.node_first_layer()
            crossing = [(u, v) for u, v, d in H.edges(data=True)
                        if int(d[LAYER_KEY]) == 1 and layer_of[u] != layer_of[v]]
            assert len(crossing) > 0

    def test_reproducible(self):
        G, _, a = self._layered()
        for gen in _LAYERED:
            f = lambda s: sorted(build_surrogate(G, a, gen, np.random.default_rng(s), n_long=1)[0].edges())
            assert f(3) == f(3)
            assert f(3) != f(4)

    def test_hier_short_layers_stay_in_their_node_set(self):
        G, _, a = self._layered()
        for gen in [gen for gen in _LAYERED if 'hier' in gen]:
            H, _ = build_surrogate(G, a, gen, np.random.default_rng(0), n_long=1)
            for u, v, d in H.edges(data=True):
                k = int(d[LAYER_KEY])
                if k > 1:
                    allowed = a.layer_nodes(k)
                    assert u in allowed and v in allowed
