"""Tests for filtrations, persistence and diagram distances."""

import numpy as np
import pytest

from topology import (
    betti_curve,
    betti_distances,
    build_simplex_tree,
    bottleneck_distance,
    diagram_distances,
    essential_count,
    wasserstein_distance,
)


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


INF = float("inf")


class TestBettiCurve:
    def test_counts_bars_alive(self):
        dgm = np.array([[0.0, 2.0], [1.0, 3.0]])
        x = np.array([0.5, 1.5, 2.5, 3.5])
        np.testing.assert_array_equal(betti_curve(dgm, x), [1, 2, 1, 0])

    def test_essential_bars_stay_alive_without_a_cap(self):
        dgm = np.array([[0.0, 1.0], [0.0, INF]])
        x = np.linspace(0.0, 100.0, 5)
        np.testing.assert_array_equal(betti_curve(dgm, x), [2, 1, 1, 1, 1])

    def test_empty_diagram_is_all_zero(self):
        x = np.linspace(0.0, 1.0, 4)
        np.testing.assert_array_equal(betti_curve(np.empty((0, 2)), x), [0, 0, 0, 0])
        np.testing.assert_array_equal(betti_curve(None, x), [0, 0, 0, 0])


class TestEssentialCount:
    def test_counts_infinite_deaths(self):
        assert essential_count(np.array([[0.0, 1.0], [0.0, INF], [0.0, INF]])) == 2
        assert essential_count(np.empty((0, 2))) == 0


class TestBettiDistances:
    def test_identical_diagrams_are_zero(self):
        dgms = [np.array([[0.0, 1.0], [0.0, INF]]), np.array([[0.5, 2.0]])]
        x = np.linspace(0.0, 3.0, 64)
        for value in betti_distances(dgms, dgms, x).values():
            assert value == pytest.approx(0.0, abs=1e-12)

    def test_orders_similarity(self):
        base = [np.array([[0.0, 2.0]]), np.empty((0, 2))]
        near = [np.array([[0.0, 2.2]]), np.empty((0, 2))]
        far = [np.array([[0.0, 8.0]]), np.empty((0, 2))]
        x = np.linspace(0.0, 10.0, 512)
        assert betti_distances(base, near, x)[(0, "betti_l1")] < betti_distances(base, far, x)[(0, "betti_l1")]

    def test_needs_no_cap_to_separate_essential_mismatch(self):
        """An extra essential bar must register, with no cap anywhere."""
        one = [np.array([[0.0, INF]]), np.empty((0, 2))]
        three = [np.array([[0.0, INF], [0.0, INF], [0.0, INF]]), np.empty((0, 2))]
        x = np.linspace(0.0, 4.0, 256)
        out = betti_distances(one, three, x)
        assert out[(0, "betti_l1")] == pytest.approx(8.0, rel=1e-2)   # 512 * (4/255 intervals)

    def test_rejects_degenerate_grid(self):
        d = [np.array([[0.0, 1.0]]), np.empty((0, 2))]
        with pytest.raises(ValueError):
            betti_distances(d, d, np.array([0.0]))
