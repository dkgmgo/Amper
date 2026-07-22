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
        kmeans = KMeans(n_clusters=self.n_components, n_init=10, random_state=self.random_state).fit(X.reshape(-1, 1))
        labels = kmeans.labels_

        self.weights_ = np.zeros(self.n_components)
        self.alphas_ = np.zeros(self.n_components)
        self.betas_ = np.zeros(self.n_components)
        self.means_ = np.zeros(self.n_components)

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


def select_components(X: Iterable[float], n_range: range = None, max_iter: int = 1000, n_init=5, seed: int = 0) -> tuple[GammaMixture, list]:
    """
    Selects the best number of mixture components based on BIC scores
    """
    #TODO include n_init in model selection
    n_range = list(n_range)
    if len(n_range) == 1:
        log.info(f"Using fixed n_components = {n_range[0]} ...")
    else:
        log.info(f"Selecting the best n_components in {n_range} ...")
    
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
