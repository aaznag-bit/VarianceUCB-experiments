from __future__ import annotations
import os
os.environ.setdefault('NUMBA_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta, t as student_t
from empirical import prepare_population, run_empirical
from simulation import _largest_remainder
DEPARTMENT_CODES = [f'HT{i:02d}' for i in range(1, 11)]
M = 20000
K = 1224
BROAD = [195, 198, 395, 494, 987, 1481, 1974, 2961, 3948]
BUDGETS = sorted(set(BROAD + list(range(423, 551))))
PROTOCOL = 'section8-refined-2x-3x-fixed-20000-v1'

def _round_numbers(series):
    text = series.astype('string').str.strip()
    number = pd.to_numeric(text, errors='coerce')
    labelled = text.str.extract('^(?:round\\s*|r\\s*)([345])$', flags=re.I, expand=False)
    return number.fillna(pd.to_numeric(labelled, errors='coerce'))

def _normal_name(value):
    if pd.isna(value):
        return ''
    text = unicodedata.normalize('NFKD', str(value).strip().casefold())
    return re.sub('[^a-z0-9]', '', ''.join((c for c in text if not unicodedata.combining(c))))

def _departments(frame):
    codes = frame['adm1_pcode'].astype('string').str.strip().str.upper()
    names = frame['adm1_name'].astype('string').str.strip()
    valid = codes.isin(DEPARTMENT_CODES)
    lookup = {}
    for (code, name) in zip(codes[valid], names[valid]):
        normalized = _normal_name(name)
        if normalized:
            lookup.setdefault(normalized, set()).add(str(code))
    filled = 0
    for index in frame.index[~valid]:
        mapping = lookup.get(_normal_name(names.loc[index]), set())
        if len(mapping) == 1:
            codes.loc[index] = next(iter(mapping))
            filled += 1
    return (codes, names, filled)

def seed_for(T):
    digest = hashlib.sha256(f'{PROTOCOL}-new-independent-stream-T{T}'.encode()).hexdigest()
    return (20261021 + int(digest[:8], 16)) % (2 ** 31 - 1)

def load_population(path):
    with Path(path).open('rb') as stream:
        sample = stream.read(16384)
    encoding = 'utf-16' if sample[:2] in (bytes([255, 254]), bytes([254, 255])) else 'utf-8-sig'
    try:
        delimiter = csv.Sniffer().sniff(sample.decode(encoding, errors='replace'), delimiters=',;\t|').delimiter
    except csv.Error:
        delimiter = ','
    frame = pd.read_csv(path, encoding=encoding, sep=delimiter, low_memory=False)
    required = {'adm0_iso3', 'round', 'adm1_pcode', 'adm1_name', 'hdds_score', 'weight_final'}
    if not required.issubset(frame.columns):
        raise ValueError('Missing columns: ' + ', '.join(sorted(required - set(frame.columns))))
    country = frame['adm0_iso3'].astype('string').str.strip().str.upper()
    rounds = _round_numbers(frame['round'])
    if len(frame) != 1974 or not country.eq('HTI').fillna(False).all() or (not rounds.eq(5).all()):
        raise ValueError('Expected the 1974-record Haiti Round 5 household file')
    (codes, names, _) = _departments(frame)
    y = pd.to_numeric(frame['hdds_score'], errors='coerce').to_numpy(dtype=float)
    w = pd.to_numeric(frame['weight_final'], errors='coerce').to_numpy(dtype=float)
    geography = codes.isin(DEPARTMENT_CODES).to_numpy(dtype=bool)
    usable = geography & np.isfinite(y) & (y >= 0) & (y <= 12) & np.isfinite(w) & (w > 0)
    if usable.sum() != 1961 or geography.sum() != 1961:
        raise ValueError('Expected 1961 eligible records after departmental exclusions')
    (values, weights, departments, realized) = ([], [], [], [])
    for code in DEPARTMENT_CODES:
        domain = codes.eq(code).fillna(False).to_numpy(dtype=bool)
        keep = domain & usable
        if keep.sum() < 2:
            raise ValueError('Fewer than two eligible households in ' + code)
        values.append(y[keep, None])
        raw_weights = w[keep].copy()
        weights.append(raw_weights / raw_weights.sum())
        labels = names[domain].dropna()
        departments.append(str(labels.value_counts().index[0]) if len(labels) else code)
        realized.append(int(domain.sum()))
    population = prepare_population(values, weights, departments, ('hdds',))
    if not np.all(population.variances[:, 0] > 0):
        raise ValueError('Every departmental variance must be positive')
    realized = np.asarray(realized, dtype=int)
    return (population, realized / realized.sum())

def statistics(values):
    values = np.asarray(values, float)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    return (mean, se, mean - 1.959963984540054 * se, mean + 1.959963984540054 * se)

def summarize(population, T, policy, result):
    n = result['counts']
    mu = result['means'][:, :, 0]
    samplevar = result['samplevar'][:, :, 0]
    variance = population.variances[:, 0]
    truth = population.means[:, 0]
    errors = mu - truth[None, :]
    critical = np.full(T + 1, np.nan)
    critical[2:] = student_t.ppf(0.975, np.arange(1, T))
    halfwidth = critical[n] * np.sqrt(samplevar / n)
    covered = np.abs(errors) <= halfwidth
    delta = np.max(1 - n / (T * variance[None, :] / variance.sum()), axis=1)
    proxy = 1.96 * np.sqrt(variance[None, :] / n).max(axis=1)
    successes = int((proxy <= 0.5).sum())
    count = len(n)
    lower = float(beta.ppf(0.05 / K, successes, count - successes + 1)) if successes else 0.0
    row = dict(T=T, policy=policy, M=count, seed=seed_for(T), max_department_mse=float((errors ** 2).mean(axis=0).max()), average_department_coverage=float(covered.mean()), joint_department_coverage=float(covered.all(axis=1).mean()), delta_p95=float(np.quantile(delta, 0.95)), delta_p99=float(np.quantile(delta, 0.99)), successes=successes, success_probability=successes / count, simultaneous_lower=lower)
    for (name, values) in [('worst_student_halfwidth', halfwidth.max(axis=1)), ('delta_max', delta), ('proxy_halfwidth', proxy)]:
        (mean, se, lo, hi) = statistics(values)
        row.update({name: mean, name + '_mcse': se, name + '_ci_lo': lo, name + '_ci_hi': hi})
    departments = [dict(T=T, policy=policy, department=population.department_names[g], mse=float(np.mean(errors[:, g] ** 2)), bias=float(errors[:, g].mean()), coverage=float(covered[:, g].mean()), mean_count=float(n[:, g].mean())) for g in range(len(variance))]
    return (row, departments)

def run_budget(population, reference, T):
    sd = np.sqrt(population.variances[:, 0])
    results = {}
    for (factor, policies) in [(2, ['vucb']), (3, ['vucb', 'static_reference', 'integer_oracle'])]:
        draws = run_empirical(population, T, policies, M=M, seed=seed_for(T), c=factor * sd, reference_weights=reference)
        for (policy, result) in draws.items():
            results[f'vucb_{factor}x' if policy == 'vucb' else policy] = result
    return results

def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

def thresholds(population, reference, rows):
    var = population.variances[:, 0]
    G = len(var)
    required = np.maximum(2, np.ceil(1.96 ** 2 * var / 0.5 ** 2)).astype(int)
    T = 2 * G
    while np.any(2 + _largest_remainder(reference, T - 2 * G) < required):
        T += 1
    output = {'integer_oracle': int(required.sum()), 'static_reference': T}
    for policy in ('vucb_2x', 'vucb_3x'):
        passing = [r for r in rows if r['policy'] == policy and r['simultaneous_lower'] >= 0.95]
        first = min(passing, key=lambda r: r['T']) if passing else None
        output[policy] = None if first is None else dict(T=first['T'], success_probability=first['success_probability'], simultaneous_lower=first['simultaneous_lower'], savings_vs_realized=1 - first['T'] / T)
    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('round5_csv', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'results' / 'section8')
    args = parser.parse_args()
    (population, reference) = load_population(args.round5_csv)
    fingerprint = hashlib.sha256(args.round5_csv.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    (rows, departments) = ([], [])
    for T in BUDGETS:
        path = args.output / f'T{T:06d}.json'
        raw = path.with_suffix('.npz')
        if path.exists() and raw.exists():
            record = json.loads(path.read_text())
            if record['source_sha256'] != fingerprint or record['M'] != M or record['K'] != K:
                raise ValueError('Saved results use a different input or protocol')
        else:
            results = run_budget(population, reference, T)
            (result_rows, result_departments) = ([], [])
            for (policy, result) in results.items():
                (row, dept) = summarize(population, T, policy, result)
                result_rows.append(row)
                result_departments.extend(dept)
            arrays = {policy + '__' + key: value for (policy, result) in results.items() for (key, value) in result.items()}
            np.savez_compressed(raw, **arrays)
            record = dict(T=T, M=M, K=K, source_sha256=fingerprint, summary=result_rows, departments=result_departments)
            path.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
        rows.extend(record['summary'])
        departments.extend(record['departments'])
        print(T, flush=True)
    write_csv(args.output / 'summary.csv', rows)
    write_csv(args.output / 'departments.csv', departments)
    (args.output / 'thresholds.json').write_text(json.dumps(thresholds(population, reference, rows), indent=2) + '\n')
if __name__ == '__main__':
    main()
