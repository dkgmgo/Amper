"""
K-component Gamma mixture fitted by EM, with BIC model selection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.special import gammaln, logsumexp, psi
from scipy.stats import gamma
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.optimize import fsolve

log = logging.getLogger(__name__)


class GammaMixture:
    def __init__(self, n_components=2, max_iter=50, random_state=123):
        self.n_components = n_components
        self.random_state = random_state
        self.max_iter = max_iter
        self.rng_ = np.random.default_rng(self.random_state)
        self.ll_history_ = []
        self.converged_ = False

    def fit(self, X):
        X = np.asarray(X).ravel()
        if np.any(X <= 0):
            raise ValueError(f"Gamma mixture requires strictly positive data. Got {(X <= 0).sum()} non-positive values.")
        
        self._initialize(X)
        prev_ll = -np.inf

        for i in range(self.max_iter):
            self._e_step()
            self._m_step()
            ll = np.sum(logsumexp(self._score(), axis=1))
            self.ll_history_.append(ll)
            if abs(ll - prev_ll) < 1e-4:
                self.converged_ = True
                log.info(f"Terminated at iteration {i+1}")
                break
            prev_ll = ll
        
        self.means_ = self.alphas_/self.betas_
        
        return self
    
    def aic(self, X):
        return 2*self._n_parameters() - 2*np.sum(logsumexp(self._score(X), axis=1))

    def bic(self, X):
        if X is None:
            N = self.X_.shape[0]
        else:
            N = len(X)
        return self._n_parameters() * np.log(N) - 2*np.sum(logsumexp(self._score(X), axis=1))
    
    def predict_proba(self, X): #responsabilities
        X = np.asarray(X).ravel()
        return self._e_step(X, fitting=False)
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
    
    def _initialize(self, X):
        #TODO maybe scale it
        kmeans = KMeans(n_clusters=self.n_components, n_init=10).fit(X.reshape(-1, 1))
        labels = kmeans.labels_

        self.weights_ = np.zeros(self.n_components)
        self.alphas_ = np.zeros(self.n_components)
        self.betas_ = np.zeros(self.n_components)

        for i in range(self.n_components):
            cluster_data = X[labels == i]
            self.weights_[i] = len(cluster_data) / len(X)
            if len(cluster_data) > 1:
                m = np.mean(cluster_data)
                v = np.var(cluster_data)
                self.betas_[i] = m / v if v > 0 else 1.0
                self.alphas_[i] = m * self.betas_[i]
            else:
                self.betas_[i] = 1.0
                self.alphas_[i] = 1.0

        self.X_ = X
        self.log_X_ = np.log(X)

    def _score(self, X=None):
        if X is None:
            X = self.X_
            log_X = self.log_X_
        else:
            log_X = np.log(X)

        log_norm = self.alphas_ * np.log(self.betas_) - gammaln(self.alphas_)
        log_dist = log_norm + np.outer(log_X, self.alphas_ -1) - np.outer(X, self.betas_)
        log_joint = np.log(self.weights_) + log_dist

        return log_joint

    def _e_step(self, X=None, fitting=True):
        log_joint = self._score(X)
        log_p = log_joint - logsumexp(log_joint, axis=1, keepdims=True)
        p = np.exp(log_p)
        if fitting:
            self.p_ = p
        return p

    def _m_step(self):
        for i in range(self.n_components):
            sum_p = np.sum(self.p_[:, i])
            if sum_p <= 1e-12: continue

            self.weights_[i] = sum_p / len(self.X_)

            log_E_X = np.log(np.sum(self.p_[:, i] * self.X_) / sum_p)
            E_log_X = np.sum(self.p_[:, i] * np.log(self.X_ + 1e-12)) / sum_p
            target = log_E_X - E_log_X

            self.alphas_[i] = fsolve(self._alpha_eq, x0=self.alphas_[i], args=(target,))[0]

            self.betas_[i] = (sum_p * self.alphas_[i]) / (np.sum(self.p_[:, i] * self.X_) + 1e-12)

    def _alpha_eq(self, a, target):
        a = max(a, 1e-6)
        return np.log(a) - psi(a) - target
    
    def _n_parameters(self):
        return 3*self.n_components - 1
    
    def sample(self, n_samples, with_components=False):
        components = self.rng_.choice(self.n_components, size=n_samples, p=self.weights_)
        samples = np.array([
            gamma.rvs(self.alphas_[k], scale=1.0 / self.betas_[k])
            for k in components
        ])

        if with_components:
            return samples, components

        return samples


def select_components(X: Iterable[float], n_range: range = None, fixed: int = None, max_iter: int = 1000, seed: int = 0) -> tuple[GammaMixture, list]:
    """
    Selects the best number of mixture components based on BIC scores
    """
    if fixed is not None:
        log.info(f"Using fixed n_components = {fixed} ...")
        gmm = GammaMixture(n_components=fixed, max_iter=max_iter, random_state=seed)
        gmm.fit(X)
        return gmm, None
    
    log.info(f"Selecting the best n_components in {list(n_range)} ...")
    bics, models = [], []
    for k in tqdm(n_range):
        gmm = GammaMixture(n_components=k, max_iter=max_iter, random_state=seed)
        gmm.fit(X)
        bics.append(gmm.bic(X))
        models.append(gmm)
    best_index = np.argmin(bics)
    best_k = list(n_range)[best_index]
    log.info(f"Best n_components by BIC: {best_k}")
    return models[best_index], bics

# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_bic_curve(bics: dict[int, float], n_range: range, path: str | Path) -> None:
    """Save the BIC-vs-K curve, marking the selected K."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6, 4))
    ks = list(n_range)
    ax.plot(ks, bics, "o-", label="BIC", linewidth=2, color="blue")
    best = ks[np.argmin(bics)]
    ax.axvline(best, color="red", linestyle="--", label=f"selected K={best}")
    ax.set_xlabel("number of components K")
    ax.set_ylabel("BIC")
    ax.set_title("Gamma mixture model selection")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_mixture_fit(X: Iterable[float], model: GammaMixture, path: str | Path, bins: int = 60) -> None:
    """Save the fitted mixture density over the empirical histogram."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    X = np.array(X)

    # histogram
    ax.hist(X, bins=bins, density=True, alpha=0.8, color="lightgray", edgecolor="white", label="Edge Distances")

    x = np.linspace(X.min(), X.max(), 512)
    pdfs = []
    for k in range(model.n_components):
        alpha = model.alphas_[k]
        beta = model.betas_[k]
        pdfs.append(gamma.pdf(x, a=alpha, scale=1/beta))

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
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
