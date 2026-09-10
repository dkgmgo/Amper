"""Tests for the Gamma mixture EM and BIC selection."""

import numpy as np
import pytest

from mixture import GammaMixture, select_components


def _sorted_by_mean(model: GammaMixture):
    order = np.argsort(model.means_)
    return model.weights_[order], model.alphas_[order], 1/model.betas_[order], model.means_[order]


def _synthetic(rng, specs):
    """specs: list of (shape, scale, n). Returns shuffled samples and labels."""
    xs, labels = [], []
    for i, (a, theta, n) in enumerate(specs):
        xs.append(rng.gamma(a, theta, size=n))
        labels.append(np.full(n, i))
    x = np.concatenate(xs)
    y = np.concatenate(labels)
    perm = rng.permutation(x.size)
    return x[perm], y[perm]


class TestParameterRecovery:
    def test_two_components(self):
        rng = np.random.default_rng(42)
        # Means 1.0 and 15.0 — well separated.
        x, y = _synthetic(rng, [(2.0, 0.5, 3000), (30.0, 0.5, 3000)])
        model = select_components(X=x, n_range=range(2,3), seed=0)[0]
        assert model.converged_
        weights, shapes, scales, means = _sorted_by_mean(model)
        np.testing.assert_allclose(means, [1.0, 15.0], rtol=0.10)
        np.testing.assert_allclose(weights, [0.5, 0.5], atol=0.05)
        np.testing.assert_allclose(shapes, [2.0, 30.0], rtol=0.30)
        np.testing.assert_allclose(scales, [0.5, 0.5], rtol=0.10)

    def test_two_components_classification(self):
        rng = np.random.default_rng(7)
        x, y = _synthetic(rng, [(2.0, 0.5, 2000), (30.0, 0.5, 2000)])
        model = select_components(X=x, n_range=range(2,3), seed=0)[0]
        pred = model.predict(x)
        # Align predicted component index with true label via mean order.
        order = np.argsort(model.means_)
        remap = np.empty(2, dtype=int)
        remap[order] = np.arange(2)
        accuracy = (remap[pred] == y).mean()
        assert accuracy > 0.95

    def test_three_components(self):
        rng = np.random.default_rng(3)
        # Means 1.0, 8.0, 48.0.
        x, _ = _synthetic(rng, [(2.0, 0.5, 3000), (20.0, 0.4, 3000), (60.0, 0.8, 3000)])
        model = select_components(X=x, n_range=range(3, 4), seed=0)[0]
        _, _, _, means = _sorted_by_mean(model)
        np.testing.assert_allclose(means, [1.0, 8.0, 48.0], rtol=0.15)


class TestBicSelection:
    def test_selects_two_on_bimodal_data(self):
        rng = np.random.default_rng(11)
        x, _ = _synthetic(rng, [(2.0, 0.5, 2000), (30.0, 0.5, 2000)])
        model, bics = select_components(X=x, n_range=range(1, 5), seed=0)
        print()
        print(bics)
        assert np.argmin(bics) == 1
        assert model.n_components == 2
        assert bics[2] > bics[1] < bics[0] #coude

    def test_k1_degenerate_ok(self):
        rng = np.random.default_rng(5)
        x = rng.gamma(4.0, 2.0, size=500)
        model, _ = select_components(X=x, n_range=[1], seed=0)
        assert model.n_components == 1
        np.testing.assert_allclose(model.means_[0], 8.0, rtol=0.15)


class TestRobustness:
    def test_nonpositive_values_raise_error(self):
        rng = np.random.default_rng(0)
        x = np.concatenate([rng.gamma(5.0, 1.0, size=500), [0.0, 0.0]])
        with pytest.raises(ValueError):
            select_components(X=x, n_range=range(1, 2), seed=0)[0]

    def test_responsibilities_sum_to_one(self):
        rng = np.random.default_rng(1)
        x = rng.gamma(3.0, 1.0, size=300)
        model = select_components(X=x, n_range=range(2,3), seed=0)[0]
        resp = model.predict_proba(x)
        assert resp.shape == (300, 2)
        np.testing.assert_allclose(resp.sum(axis=1), 1.0)

    def test_bic_penalises_parameters(self):
        rng = np.random.default_rng(2)
        x = rng.gamma(3.0, 1.0, size=1000)
        m1 = select_components(X=x, n_range=range(1,2), seed=0)[0]
        m3 = select_components(X=x, n_range=range(3, 4), seed=0)[0]
        # On single-component data, BIC must prefer K=1.
        assert m1.bic(x) < m3.bic(x)


class TestSampling:
    """`GammaMixture.sample` must be driven only by `random_state`."""

    def _fitted(self, seed=0):
        rng = np.random.default_rng(123)
        x, _ = _synthetic(rng, [(2.0, 0.5, 1000), (30.0, 0.5, 1000)])
        return select_components(X=x, n_range=range(2, 3), seed=seed)[0]

    def test_same_random_state_gives_same_samples(self):
        a = self._fitted()
        b = self._fitted()
        np.testing.assert_array_equal(a.sample(50), b.sample(50))

    def test_samples_ignore_global_numpy_seed(self):
        np.random.seed(1234)
        first = self._fitted().sample(50)
        np.random.seed(9999)
        second = self._fitted().sample(50)

        np.testing.assert_array_equal(first, second)

    def test_with_components_matches_component_of_each_sample(self):
        model = self._fitted()
        samples, components = model.sample(4000, with_components=True)

        assert samples.shape == components.shape == (4000,)
        assert set(np.unique(components)) <= {0, 1}
        for k in range(model.n_components):
            drawn = samples[components == k]
            if drawn.size > 100:
                expected = model.alphas_[k] / model.betas_[k]
                np.testing.assert_allclose(drawn.mean(), expected, rtol=0.15)

    def test_sample_mean_matches_mixture_mean(self):
        model = self._fitted()
        expected = float(np.sum(model.weights_ * model.means_))
        np.testing.assert_allclose(model.sample(20000).mean(), expected, rtol=0.10)


class TestPomegranateComparison:
    """Compare our EM against pomegranate's Gamma mixture"""

    def test_against_pomegranate(self):
        pomegranate = pytest.importorskip("pomegranate")
        import torch
        from pomegranate.distributions import Gamma
        from pomegranate.gmm import GeneralMixtureModel

        rng = np.random.default_rng(42)
        x, _ = _synthetic(rng, [(2.0, 0.5, 3000), (30.0, 0.5, 3000)])

        ours = fit_gamma_mixture(x, 2, seed=0)
        pom = GeneralMixtureModel([Gamma(), Gamma()], verbose=False)
        pom.fit(torch.tensor(x[:, None], dtype=torch.float64))
        pom_means = np.sort(
            np.array([(d.shapes / d.rates).item() for d in pom.distributions])
        )
        np.testing.assert_allclose(np.sort(ours.means), pom_means, rtol=0.15)
