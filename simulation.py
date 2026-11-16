from __future__ import annotations
import math
import numpy as np
from numba import njit, prange

@njit(inline='always')
def _mix64(x):
    x = np.uint64(x) + np.uint64(11400714819323198485)
    x = (x ^ x >> np.uint64(30)) * np.uint64(13787848793156543929)
    x = (x ^ x >> np.uint64(27)) * np.uint64(10723151780598845931)
    return x ^ x >> np.uint64(31)

@njit(inline='always')
def _open_uniform(z):
    u = (float(z >> np.uint64(11)) + 0.5) * (1.0 / 9007199254740992.0)
    return min(u, 1.0 - 1.0 / 9007199254740992.0)

@njit(inline='always')
def _uniform(key, index):
    z = _mix64(key + np.uint64(index) * np.uint64(11400714819323198485))
    return _open_uniform(z)

@njit(inline='always')
def _key(seed, replication, group):
    return _mix64(seed ^ _mix64(np.uint64(replication + 1)) ^ _mix64(np.uint64(group + 1) + np.uint64(15111065706836454659)))

@njit(inline='always')
def _draw(key, index, family, sigma, normal_cache, group):
    if family == 0:
        if index % 2 == 1:
            return sigma * normal_cache[group]
        u = _uniform(key, index)
        v = _uniform(key, index + 1)
        radius = math.sqrt(-2.0 * math.log(u))
        angle = 2.0 * math.pi * v
        normal_cache[group] = radius * math.sin(angle)
        return sigma * radius * math.cos(angle)
    u = _uniform(key, index)
    if family == 1:
        return math.sqrt(3.0) * sigma * (2.0 * u - 1.0)
    return -sigma * math.log(u)

@njit(inline='always')
def _observe(g, family, sigmas, keys, counts, means, m2, cache):
    y = _draw(keys[g], counts[g], family, sigmas[g], cache, g)
    counts[g] += 1
    delta = y - means[g]
    means[g] += delta / counts[g]
    m2[g] += delta * (y - means[g])

@njit(inline='always')
def _better(g, h, scores):
    return scores[g] > scores[h] or (scores[g] == scores[h] and g < h)

@njit(inline='always')
def _heap_down(heap, scores, pos):
    n = len(heap)
    while True:
        child = 2 * pos + 1
        if child >= n:
            break
        if child + 1 < n and _better(heap[child + 1], heap[child], scores):
            child += 1
        if not _better(heap[child], heap[pos], scores):
            break
        (heap[pos], heap[child]) = (heap[child], heap[pos])
        pos = child

@njit
def _heap_build(scores):
    heap = np.arange(len(scores), dtype=np.int64)
    for pos in range(len(heap) // 2 - 1, -1, -1):
        _heap_down(heap, scores, pos)
    return heap

@njit(inline='always')
def _sd(g, counts, m2):
    return math.sqrt(max(0.0, m2[g] / (counts[g] - 1)))

@njit(inline='always')
def _ucb(g, counts, means, m2, ucb_kind, c, rootlog, logterm, bas_width, true_sigma=0.0):
    n = counts[g]
    if ucb_kind == 0:
        return _sd(g, counts, m2) * (1.0 + rootlog[n] + logterm[n])
    if ucb_kind == 1:
        return _sd(g, counts, m2) + c[g] * rootlog[n]
    if ucb_kind == 2:
        return means[g] * (1.0 + rootlog[n])
    if ucb_kind == 4:
        sample_sd = _sd(g, counts, m2)
        gaussian = sample_sd * (1.0 + rootlog[n] + logterm[n])
        calibrated = c[g]
        if abs(gaussian - true_sigma) <= abs(calibrated - true_sigma):
            return gaussian
        return calibrated
    return _sd(g, counts, m2) + bas_width[n]

@njit(inline='always')
def _index(u, n, q):
    if q == 2.0:
        return u * u / n
    if q == 1.0:
        return u / n
    return u ** q / n

@njit
def _largest_remainder(weights, total):
    G = len(weights)
    result = np.zeros(G, dtype=np.int64)
    denominator = weights.sum()
    if denominator <= 0.0:
        denominator = float(G)
        weights = np.ones(G)
    fractional = np.empty(G)
    for g in range(G):
        quota = total * weights[g] / denominator
        result[g] = int(math.floor(quota))
        fractional[g] = quota - result[g]
    for _ in range(total - result.sum()):
        best = 0
        for g in range(1, G):
            if fractional[g] > fractional[best]:
                best = g
        result[best] += 1
        fractional[best] = -1.0
    return result

@njit
def _oracle_finite(sigmas, T, p, min_count):
    counts = np.full(len(sigmas), min_count, dtype=np.int64)
    scores = np.empty(len(sigmas))
    weights = sigmas ** (2.0 * p)
    for g in range(len(sigmas)):
        scores[g] = weights[g] * (counts[g] ** (-p) - (counts[g] + 1) ** (-p))
    heap = _heap_build(scores)
    for _ in range(T - counts.sum()):
        g = heap[0]
        counts[g] += 1
        scores[g] = weights[g] * (counts[g] ** (-p) - (counts[g] + 1) ** (-p))
        _heap_down(heap, scores, 0)
    return counts

def integer_oracle(sigmas, T, p, min_count=2):
    sigmas = np.asarray(sigmas, dtype=float)
    T = int(T)
    if T < len(sigmas) * min_count or min_count < 1:
        raise ValueError('Budget cannot meet the minimum sample counts')
    if np.any(sigmas < 0) or not np.all(np.isfinite(sigmas)):
        raise ValueError('Invalid standard deviations')
    if math.isfinite(p):
        if p < 1:
            raise ValueError('p must be at least one')
        return _oracle_finite(sigmas, T, float(p), int(min_count))
    variance = sigmas * sigmas
    if variance.max(initial=0.0) == 0:
        return np.full(len(sigmas), min_count, dtype=np.int64) + _largest_remainder(np.ones(len(sigmas)), T - len(sigmas) * min_count)
    (lo, hi) = (0.0, float(variance.max() / min_count))
    for _ in range(90):
        level = (lo + hi) / 2.0
        needed = np.maximum(min_count, np.ceil(variance / level)).sum()
        if needed <= T:
            hi = level
        else:
            lo = level
    counts = np.maximum(min_count, np.ceil(variance / hi)).astype(np.int64)
    remaining = T - int(counts.sum())
    counts[:remaining] += 1
    if remaining < 0 or counts.sum() != T:
        raise ArithmeticError('Integer threshold failed to produce a feasible allocation')
    return counts

def oracle_proportions(sigmas, p):
    sigmas = np.asarray(sigmas, dtype=float)
    q = 2.0 if np.isinf(p) else 2.0 * p / (p + 1.0)
    weights = sigmas ** q
    return weights / weights.sum()

def normalized_regret(counts, sigmas, p):
    counts = np.asarray(counts)
    sigmas = np.asarray(sigmas, dtype=float)
    T = counts.sum(axis=-1)
    if np.isinf(p):
        proxy = np.max(sigmas ** 2 / counts, axis=-1)
        oracle = np.sum(sigmas ** 2) / T
    else:
        proxy = np.sum((sigmas ** 2 / counts) ** p, axis=-1) ** (1.0 / p)
        oracle = np.sum(sigmas ** (2.0 * p / (p + 1.0))) ** (1.0 + 1.0 / p) / T
    return proxy / oracle - 1.0

@njit
def _one(seed, replication, family, policy, sigmas, T, q, ucb_kind, c_base, alpha, static_counts, rootlog, logterm, bas_width):
    G = len(sigmas)
    keys = np.empty(G, dtype=np.uint64)
    counts = np.zeros(G, dtype=np.int64)
    means = np.zeros(G)
    m2 = np.zeros(G)
    cache = np.zeros(G)
    for g in range(G):
        keys[g] = _key(seed, replication, g)
    initial = max(2, int(math.ceil(T / (2.0 * G)))) if policy == 1 else 2
    for g in range(G):
        for _ in range(initial):
            _observe(g, family, sigmas, keys, counts, means, m2, cache)
    c = c_base.copy() * alpha
    calibration_fail = False
    if ucb_kind == 1 or ucb_kind == 4:
        for g in range(G):
            calibration_fail = calibration_fail or c[g] < sigmas[g]
    failure = False
    if policy == 0:
        for t in range(initial * G, T):
            _observe(t % G, family, sigmas, keys, counts, means, m2, cache)
    elif policy == 1:
        weights = np.empty(G)
        for g in range(G):
            weights[g] = _sd(g, counts, m2) ** q
        extra = _largest_remainder(weights, T - counts.sum())
        for g in range(G):
            for _ in range(extra[g]):
                _observe(g, family, sigmas, keys, counts, means, m2, cache)
    elif policy == 4:
        for g in range(G):
            while counts[g] < static_counts[g]:
                _observe(g, family, sigmas, keys, counts, means, m2, cache)
    else:
        scores = np.empty(G)
        bounds = np.empty(G)
        for g in range(G):
            bounds[g] = _ucb(g, counts, means, m2, ucb_kind, c, rootlog, logterm, bas_width, sigmas[g])
            failure = failure or bounds[g] < sigmas[g]
            scores[g] = _index(bounds[g], counts[g], q)
        heap = _heap_build(scores)
        for t in range(counts.sum(), T):
            g = heap[0]
            _observe(g, family, sigmas, keys, counts, means, m2, cache)
            bounds[g] = _ucb(g, counts, means, m2, ucb_kind, c, rootlog, logterm, bas_width, sigmas[g])
            failure = failure or bounds[g] < sigmas[g]
            scores[g] = _index(bounds[g], counts[g], q)
            _heap_down(heap, scores, 0)
    variance = np.empty(G)
    for g in range(G):
        variance[g] = max(0.0, m2[g] / (counts[g] - 1))
    return (counts, means, variance, failure, calibration_fail, c)

@njit(parallel=True, cache=True)
def _many(seed, M, family, policy, sigmas, T, q, ucb_kind, c_base, alpha, static_counts, rootlog, logterm, bas_width):
    G = len(sigmas)
    counts = np.empty((M, G), dtype=np.int64)
    means = np.empty((M, G))
    variances = np.empty((M, G))
    failures = np.zeros(M, dtype=np.bool_)
    calibration = np.zeros(M, dtype=np.bool_)
    c_values = np.empty((M, G))
    for m in prange(M):
        (n, mean, var, failure, cal_fail, c) = _one(seed, m, family, policy, sigmas, T, q, ucb_kind, c_base, alpha, static_counts, rootlog, logterm, bas_width)
        counts[m] = n
        means[m] = mean
        variances[m] = var
        failures[m] = failure
        calibration[m] = cal_fail
        c_values[m] = c
    return (counts, means, variances, failures, calibration, c_values)
FAMILIES = {'gaussian': 0, 'uniform': 1, 'exponential': 2}
POLICIES = {'uniform': 0, 'etc': 1, 'vucb': 2, 'bas': 3, 'integer_oracle': 4}
UCBS = {'gaussian': 0, 'subgaussian': 1, 'exponential': 2, 'bas': 3, 'gaussian_closest': 4}

def run_experiment(sigmas, T, p, family, policies, M, seed, c, alpha=1.0, ucb_kind=None, confidence_power=3.5):
    sigmas = np.ascontiguousarray(sigmas, dtype=np.float64)
    (T, M, G) = (int(T), int(M), len(sigmas))
    if T < 2 * G or M < 1 or p < 1 or (not np.all(np.isfinite(sigmas) & (sigmas > 0))):
        raise ValueError('Require positive sigmas, M >= 1, p >= 1 and T >= 2G')
    family_id = FAMILIES[family]
    q = 2.0 if np.isinf(p) else 2.0 * p / (p + 1.0)
    default_ucb = ('gaussian', 'subgaussian', 'exponential')[family_id]
    c_base = np.broadcast_to(np.asarray(c, dtype=float), (G,)).copy()
    if not np.all(np.isfinite(c_base) & (c_base >= 0)):
        raise ValueError('Invalid confidence coefficients')
    n_values = np.arange(T + 1, dtype=float)
    n_values[0] = 1.0
    rootlog = np.sqrt(3.0 * np.log(T) / n_values)
    logterm = 3.0 * np.log(T) / n_values
    log_delta = -float(confidence_power) * math.log(T)
    (c1, c2) = (200.0 if family_id == 0 else 600.0, 2.0)
    b = 4.0 * math.sqrt(c1 * (math.log(c2) - log_delta)) * math.sqrt(math.log(2.0) - log_delta) + 2.0 * math.sqrt(5.0 * c1) / math.sqrt(T)
    bas_width = b / np.sqrt(n_values)
    result = {}
    for name in policies:
        kind = 'bas' if name == 'bas' else ucb_kind or default_ucb
        if name == 'bas' and (not np.isinf(p) or family_id == 2):
            raise ValueError('B-AS requires p=infinity and bounded or Gaussian observations')
        if kind == 'gaussian_closest' and family_id != 0:
            raise ValueError('Gaussian selector requires Gaussian observations')
        static = integer_oracle(sigmas, T, p) if name == 'integer_oracle' else np.full(G, 2, dtype=np.int64)
        raw = _many(np.uint64(seed), M, family_id, POLICIES[name], sigmas, T, q, UCBS[kind], c_base, float(alpha), static, rootlog, logterm, bas_width)
        (n, means, variance, failure, calibration_fail, coefficients) = raw
        if not np.all(n.sum(axis=1) == T):
            raise ArithmeticError('Allocation does not respect the budget')
        result[name] = dict(counts=n, means=means, samplevar=variance, ucb_failure=failure, calibration_fail=calibration_fail, calibration=coefficients, regret=normalized_regret(n, sigmas, p))
    return result
