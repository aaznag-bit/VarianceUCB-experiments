from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import math
import numpy as np
from numba import njit, prange
from simulation import _key, _uniform, _heap_build, _heap_down, _largest_remainder, integer_oracle

@dataclass(frozen=True)
class PackedPopulation:
    values: np.ndarray
    cumulative_weights: np.ndarray
    offsets: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    valid_weight_mass: np.ndarray
    department_names: tuple
    outcome_names: tuple

def prepare_population(values: Sequence, weights: Sequence, department_names=None, outcome_names=('hdds', 'fcg', 'duration_minutes')):
    if len(values) != len(weights) or len(values) == 0:
        raise ValueError('Require corresponding nonempty department value and weight lists')
    (arrays, cdfs) = ([], [])
    (G, K) = (len(values), len(outcome_names))
    means = np.full((G, K), np.nan)
    variances = means.copy()
    valid_mass = np.zeros((G, K))
    offsets = [0]
    for (g, (records, weight)) in enumerate(zip(values, weights)):
        y = np.asarray(records, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]
        w = np.asarray(weight, dtype=np.float64)
        if y.ndim != 2 or y.shape[1] != K or len(y) == 0 or (w.shape != (len(y),)):
            raise ValueError('Each department needs an N by K matrix and N weights')
        if not np.isfinite(y[:, 0]).all() or np.any((y[:, 0] < 0) | (y[:, 0] > 12)):
            raise ValueError('All eligible HDDS observations must be finite and in [0,12]')
        if np.any(np.isinf(y)) or not np.isfinite(w).all() or np.any(w <= 0):
            raise ValueError('Weights must be finite and positive; secondary missingness must be NaN')
        w = w / w.sum()
        cdf = np.cumsum(w)
        cdf[-1] = 1.0
        for k in range(K):
            valid = np.isfinite(y[:, k])
            mass = float(w[valid].sum())
            valid_mass[g, k] = mass
            if mass > 0:
                a = w[valid] / mass
                z = y[valid, k]
                if np.all(z == z[0]):
                    means[g, k] = z[0]
                    variances[g, k] = 0.0
                else:
                    means[g, k] = np.dot(a, z)
                    variances[g, k] = np.dot(a, (z - means[g, k]) ** 2)
        arrays.append(y)
        cdfs.append(cdf)
        offsets.append(offsets[-1] + len(y))
    names = tuple((str(g) for g in range(G))) if department_names is None else tuple(department_names)
    if len(names) != G:
        raise ValueError('Department names must match the population')
    return PackedPopulation(np.ascontiguousarray(np.vstack(arrays)), np.ascontiguousarray(np.concatenate(cdfs)), np.asarray(offsets, dtype=np.int64), means, variances, valid_mass, names, tuple(outcome_names))

@njit(inline='always')
def _record_index(key, index, g, offsets, cdf):
    u = _uniform(key, index)
    (lo, hi) = (offsets[g], offsets[g + 1])
    while lo < hi:
        mid = (lo + hi) // 2
        if u < cdf[mid]:
            hi = mid
        else:
            lo = mid + 1
    return min(lo, offsets[g + 1] - 1)

@njit(inline='always')
def _observe_empirical(g, keys, counts, observed, means, m2, values, offsets, cdf):
    row = _record_index(keys[g], counts[g], g, offsets, cdf)
    counts[g] += 1
    for k in range(values.shape[1]):
        y = values[row, k]
        if math.isfinite(y):
            observed[g, k] += 1
            delta = y - means[g, k]
            means[g, k] += delta / observed[g, k]
            m2[g, k] += delta * (y - means[g, k])

@njit(inline='always')
def _sd_primary(g, counts, m2):
    return math.sqrt(max(0.0, m2[g, 0] / (counts[g] - 1)))

@njit(parallel=True, cache=True)
def _many(seed, M, policy, T, c, static_counts, batch_size, true_sd, values, offsets, cdf, rootlog):
    (G, K) = (len(true_sd), values.shape[1])
    counts = np.empty((M, G), dtype=np.int64)
    observed = np.empty((M, G, K), dtype=np.int64)
    means = np.empty((M, G, K))
    variances = np.empty((M, G, K))
    failures = np.zeros(M, dtype=np.bool_)
    for m in prange(M):
        (n, obs, mu, var, failed) = _one(seed, m, policy, T, c, static_counts, batch_size, true_sd, values, offsets, cdf, rootlog)
        counts[m] = n
        observed[m] = obs
        means[m] = mu
        variances[m] = var
        failures[m] = failed
    return (counts, observed, means, variances, failures)

@njit
def _one(seed, replication, policy, T, c, static_counts, batch_size, true_sd, values, offsets, cdf, rootlog):
    (G, K) = (len(true_sd), values.shape[1])
    keys = np.empty(G, dtype=np.uint64)
    counts = np.zeros(G, dtype=np.int64)
    observed = np.zeros((G, K), dtype=np.int64)
    means = np.zeros((G, K))
    m2 = np.zeros((G, K))
    for g in range(G):
        keys[g] = _key(seed, replication, g)
        for _ in range(2):
            _observe_empirical(g, keys, counts, observed, means, m2, values, offsets, cdf)
    failure = False
    if policy >= 4:
        for g in range(G):
            while counts[g] < static_counts[g]:
                _observe_empirical(g, keys, counts, observed, means, m2, values, offsets, cdf)
    else:
        ucb = np.empty(G)
        scores = np.empty(G)
        for g in range(G):
            ucb[g] = _sd_primary(g, counts, m2) + c[g] * rootlog[counts[g]]
            failure = failure or ucb[g] < true_sd[g]
            scores[g] = ucb[g] ** 2 / counts[g]
        heap = _heap_build(scores)
        for t in range(2 * G, T):
            g = heap[0]
            _observe_empirical(g, keys, counts, observed, means, m2, values, offsets, cdf)
            ucb[g] = _sd_primary(g, counts, m2) + c[g] * rootlog[counts[g]]
            failure = failure or ucb[g] < true_sd[g]
            scores[g] = ucb[g] ** 2 / counts[g]
            _heap_down(heap, scores, 0)
    variances = np.full((G, K), np.nan)
    for g in range(G):
        for k in range(K):
            if observed[g, k] > 1:
                variances[g, k] = max(0.0, m2[g, k] / (observed[g, k] - 1))
            if observed[g, k] == 0:
                means[g, k] = np.nan
    return (counts, observed, means, variances, failure)

def run_empirical(population, T, policies, M, seed, c, reference_weights):
    (T, M) = (int(T), int(M))
    true_sd = np.sqrt(population.variances[:, 0])
    G = len(true_sd)
    if T < 2 * G or M < 1:
        raise ValueError('Require T >= 2G and M >= 1')
    coefficients = np.broadcast_to(np.asarray(c, float), (G,)).copy()
    rootlog = np.sqrt(3 * np.log(T) / np.maximum(1, np.arange(T + 1)))
    policy_ids = {'vucb': 2, 'integer_oracle': 4, 'static_reference': 6}
    results = {}
    for policy in policies:
        static = np.full(G, 2, dtype=np.int64)
        if policy == 'integer_oracle':
            static = integer_oracle(true_sd, T, float('inf'), min_count=2)
        elif policy == 'static_reference':
            static += _largest_remainder(np.asarray(reference_weights, float), T - 2 * G)
        (n, observed, means, variances, failures) = _many(np.uint64(seed), M, policy_ids[policy], T, coefficients, static, 50, true_sd, population.values, population.offsets, population.cumulative_weights, rootlog)
        results[policy] = dict(counts=n, observation_counts=observed, means=means, samplevar=variances, ucb_failure=failures)
    return results
