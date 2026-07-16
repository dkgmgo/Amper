"""Tests for the Gamma mixture EM and BIC selection."""

import numpy as np
import pytest

from mixture import GammaMixture, plot_bic_curve, plot_mixture_fit, select_components


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
        model = select_components(X=x, fixed=2, seed=0)[0]
        assert model.converged_
        weights, shapes, scales, means = _sorted_by_mean(model)
        np.testing.assert_allclose(means, [1.0, 15.0], rtol=0.10)
        np.testing.assert_allclose(weights, [0.5, 0.5], atol=0.05)
        np.testing.assert_allclose(shapes, [2.0, 30.0], rtol=0.30)
        np.testing.assert_allclose(scales, [0.5, 0.5], rtol=0.10)

    def test_two_components_classification(self):
        rng = np.random.default_rng(7)
        x, y = _synthetic(rng, [(2.0, 0.5, 2000), (30.0, 0.5, 2000)])
        model = select_components(X=x, fixed=2, seed=0)[0]
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
        model = select_components(X=x, fixed=3, seed=0)[0]
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
            select_components(X=x, fixed=1, seed=0)[0]

    def test_responsibilities_sum_to_one(self):
        rng = np.random.default_rng(1)
        x = rng.gamma(3.0, 1.0, size=300)
        model = select_components(X=x, fixed=2, seed=0)[0]
        resp = model.predict_proba(x)
        assert resp.shape == (300, 2)
        np.testing.assert_allclose(resp.sum(axis=1), 1.0)

    def test_bic_penalises_parameters(self):
        rng = np.random.default_rng(2)
        x = rng.gamma(3.0, 1.0, size=1000)
        m1 = select_components(X=x, fixed=1, seed=0)[0]
        m3 = select_components(X=x, fixed=3, seed=0)[0]
        # On single-component data, BIC must prefer K=1.
        assert m1.bic(x) < m3.bic(x)


class TestPlots:
    def test_plots_write_files(self, tmp_path):
        rng = np.random.default_rng(4)
        x, _ = _synthetic(rng, [(2.0, 0.5, 500), (30.0, 0.5, 500)])
        model, bics = select_components(x, range(1, 4), seed=0)
        bic_png = tmp_path / "bic.png"
        fit_png = tmp_path / "fit.png"
        plot_bic_curve(bics,range(1, 4), bic_png)
        plot_mixture_fit(x, model, fit_png)
        assert bic_png.stat().st_size > 0
        assert fit_png.stat().st_size > 0


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
