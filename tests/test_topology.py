"""Tests for filtrations, persistence and diagram distances."""

import numpy as np
import pytest

from topology import build_simplex_tree, bottleneck_distance, diagram_distances, wasserstein_distance


def square():
    """A 4-cycle, all edges at filtration value 1."""
    edges = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")]
    values = [1.0] * 4
    return edges, values


def test_identical_diagrams_have_zero_distance():
    edges, values = square()
    st = build_simplex_tree(edges, values, set([x for e in edges for x in e]))
    dgms = [st.persistence_intervals_in_dimension(i) for i in (0, 1)]
    dists = diagram_distances(dgms, dgms)
    for value in dists.values():
        assert value == pytest.approx(0.0, abs=1e-12)


def test_distance_orders_similarity():
    base = np.array([[0.0, 1.0], [0.0, 2.0]])
    near = np.array([[0.0, 1.1], [0.0, 2.1]])
    far = np.array([[0.0, 5.0], [0.0, 9.0]])
    assert bottleneck_distance(base, near) < bottleneck_distance(base, far)
    assert wasserstein_distance(base, near) < wasserstein_distance(base, far)
