"""
adagent/core.py
===============
Core building blocks for the Applicability-Domain-aware, abstaining
toxicity agent ("ADAgent"):

  features : ECFP4 (Morgan r=2, 2048-bit) fingerprints from SMILES
  splits   : random / scaffold (Bemis-Murcko) / maxmin splits
  ad       : applicability-domain scores (max & mean-topk Tanimoto to train)
  uq       : RF-ensemble uncertainty (mean prob + inter-model disagreement)
  conformal: inductive conformal prediction sets (marginal coverage guarantee)
  gate     : risk-controlled trust gate -> ANSWER / ESCALATE / ABSTAIN
  metrics  : AUROC, AUPR, ECE, risk-coverage, selective-prediction metrics

Design notes
------------
* AD score follows the leakage-audit protocol of ToxBench (BMC
  Bioinformatics 2026): nearest-neighbour Tanimoto similarity between
  test and train compounds.
* Tanimoto computed with vectorised bit-math:  T = |A&B| / |A|+|B|-|A&B|.
* Conformal prediction (Vovk et al.) gives distribution-free marginal
  coverage; ambiguous/empty prediction sets are used as abstention signals.
* The gate thresholds (AD quantile, disagreement quantile) are chosen on
  a held-out calibration split to satisfy a target answered-set error
  rate while maximising coverage ("risk-controlled answering").
"""
from __future__ import annotations

from functools import lru_cache
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

FP_BITS = 2048
_mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)

# ------------------------------------------------------------------ features
def featurize(smiles: list[str]):
    """Return (X int8 [n,2048], valid_idx). Invalid SMILES dropped."""
    rows, valid = [], []
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        fp = _mfpgen.GetFingerprint(mol)
        arr = np.zeros(FP_BITS, dtype=np.int8)
        for b in fp.GetOnBits():
            arr[b] = 1
        if arr.sum() == 0:
            continue
        rows.append(arr)
        valid.append(i)
    return np.asarray(rows, dtype=np.int8), np.asarray(valid)

# ------------------------------------------------------------------ splits
@lru_cache(maxsize=1)
def _reference_scaffolds() -> dict[str, str]:
    """Use reference scaffold keys for the bundled benchmark molecules."""
    path = Path(__file__).resolve().parent / "data" / "reference_scaffolds.json.gz"
    if not path.is_file():
        return {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _scaffold(smiles: str) -> str:
    reference = _reference_scaffolds()
    if smiles in reference:
        return reference[smiles]
    mol = Chem.MolFromSmiles(smiles)
    return "" if mol is None else MurckoScaffold.MurckoScaffoldSmiles(mol=mol)

def split_indices(n: int, smiles: list[str], method: str,
                  seed: int = 0, fracs=(0.6, 0.2, 0.2)):
    """Return (train_idx, val_idx, test_idx) as int arrays.

    random  : uniform shuffle.
    scaffold: Bemis-Murcko scaffold groups assigned wholly to one split.
    maxmin  : test set selected by farthest-point (MaxMin) sampling on
              Tanimoto distance -> chemically dissimilar test set.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    if method == "random":
        rng.shuffle(idx)
        a = int(fracs[0] * n); b = int((fracs[0] + fracs[1]) * n)
        return idx[:a], idx[a:b], idx[b:]
    if method == "scaffold":
        scaff = pd.Series([_scaffold(s) for s in smiles])
        groups = scaff.groupby(scaff).groups
        keys = list(groups.keys()); rng.shuffle(keys)
        tr, va, te = [], [], []
        for k in keys:
            g = list(groups[k])
            bucket = (tr if len(tr) / n < fracs[0]
                      else va if len(va) / n < fracs[1] else te)
            bucket.extend(g)
        return (np.array(tr, dtype=int), np.array(va, dtype=int), np.array(te, dtype=int))
    if method == "maxmin":
        tr, va, te = _maxmin_split(n, seed, fracs)
        return tr, va, te
    raise ValueError(method)

def _maxmin_split(n: int, seed: int, fracs):
    """Low-leakage maxmin split.

    Farthest-point (MaxMin) sampling proposes diverse test candidates; a
    candidate is accepted into the test set only if its maximum Tanimoto
    similarity to the still-unassigned molecules is <= 0.6. Candidates
    with close analogs stay in the unassigned pool (train/val), so no
    test molecule has a >0.6 nearest neighbour in train — a genuinely
    dissimilar test set even on analog-rich datasets.
    """
    global _SMILES
    rng = np.random.default_rng(seed)
    X, valid = featurize(_SMILES)
    m = len(valid)
    S = _tanimoto_rows(X, X)                       # full sim matrix [m, m]
    np.fill_diagonal(S, -1.0)
    n_test = int(fracs[2] * n)

    # lazy farthest-point generator (positions in X), with picked mask so
    # points are never re-selected
    def farthest_seq():
        first = int(rng.integers(m))
        picked_mask = np.zeros(m, dtype=bool)
        picked_mask[first] = True
        min_sim = S[:, first].copy()
        yield first
        while not picked_mask.all():
            cand = np.where(picked_mask, np.inf, min_sim)
            nxt = int(np.argmin(cand))
            if not np.isfinite(cand[nxt]):
                return
            picked_mask[nxt] = True
            min_sim = np.minimum(min_sim, S[:, nxt])
            yield nxt

    def component(c, unassigned_arr):
        """Transitive closure of the >0.6 similarity relation around c,
        restricted to still-unassigned molecules (BFS)."""
        comp, stack = [c], [c]
        unassigned_arr = unassigned_arr[unassigned_arr != c]
        while stack:
            t = stack.pop()
            nbrs = unassigned_arr[S[t, unassigned_arr] > 0.6]
            if len(nbrs):
                unassigned_arr = unassigned_arr[~np.isin(unassigned_arr, nbrs)]
                comp.extend(nbrs.tolist())
                stack.extend(nbrs.tolist())
        return comp, unassigned_arr

    unassigned_arr = np.arange(m)
    test_pos = []
    for c in farthest_seq():
        if len(test_pos) >= n_test:
            break
        if c not in set(unassigned_arr.tolist()):
            continue
        # transitive cluster closure: c and ALL molecules connected to it
        # by >0.6 similarity move to test together -> no analog of a test
        # molecule is left behind in train
        comp, unassigned_arr = component(c, unassigned_arr)
        if len(test_pos) + len(comp) <= n_test * 1.25:
            test_pos.extend(comp)
        else:  # component too large for the test budget: stays in train
            unassigned_arr = np.sort(np.append(unassigned_arr, comp))

    test = np.array([valid[p] for p in test_pos], dtype=int)
    rest_pos = np.array(sorted(unassigned_arr.tolist()))
    rest = valid[rest_pos]
    rng.shuffle(rest)
    n_val = int(fracs[1] * n)
    return rest[:-n_val], rest[-n_val:], test

_SMILES: list[str] = []

def set_smiles_for_splits(smiles: list[str]):
    global _SMILES
    _SMILES = smiles

# ------------------------------------------------------------------ tanimoto
def _tanimoto_rows(X: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Tanimoto similarity between each row of X and each row of `rows`.

    X: [n, bits] int8, rows: [m, bits] int8.  Returns [n, m] float32.
    """
    Xf = X.astype(np.float32)
    Rf = rows.astype(np.float32)
    dots = Xf @ Rf.T
    denom = Xf.sum(1, keepdims=True) + Rf.sum(1)[None, :] - dots
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(denom > 0, dots / denom, 0.0)
    return sim

# ------------------------------------------------------------------ AD
def ad_scores(X_train: np.ndarray, X_test: np.ndarray, topk: int = 5):
    """Nearest-neighbour Tanimoto of each test compound to the train set.

    Returns dict with 'max' and 'mean_topk' arrays (len = n_test).
    Low max-similarity == outside the applicability domain.
    """
    sims = _tanimoto_rows(X_test, X_train)          # [n_test, n_train]
    part = np.partition(sims, -topk, axis=1)[:, -topk:]
    return {"max": sims.max(1), "mean_topk": part.mean(1),
            "frac_gt06": (sims.max(1) > 0.6)}

# ------------------------------------------------------------------ UQ ensemble
def train_ensemble(X, y, n_models=3, n_trees=60, seed=0):
    """Bagged RandomForest ensemble. Returns list of fitted models."""
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(seed)
    models = []
    for m in range(n_models):
        idx = rng.choice(len(X), size=len(X), replace=True)
        uniq = np.unique(y[idx])
        if len(uniq) < 2:
            continue
        rf = RandomForestClassifier(
            n_estimators=n_trees, random_state=seed * 100 + m,
            n_jobs=2, class_weight="balanced", min_samples_leaf=3)
        rf.fit(X[idx], y[idx])
        models.append(rf)
    return models

def ensemble_predict(models, X):
    """Mean probability of class 1 and inter-model disagreement (std)."""
    P = np.asarray([m.predict_proba(X)[:, 1] for m in models])  # [M, n]
    p1 = P.mean(0)
    std = P.std(0)
    return p1, std

# ------------------------------------------------------------------ conformal
def conformal_sets(models, X_cal, y_cal, X_test, alpha=0.1):
    """Inductive conformal prediction sets at marginal coverage 1-alpha.

    Nonconformity s(x, y) = 1 - p_hat(y). A class y is included for a
    test point if 1 - p_hat(y) <= ceil((n+1)(1-alpha))/n quantile of
    calibration scores.
    """
    p_cal, _ = ensemble_predict(models, X_cal)
    p_tst, _ = ensemble_predict(models, X_test)
    s_cal = 1.0 - np.where(y_cal == 1, p_cal, 1.0 - p_cal)
    n = len(s_cal)
    k = min(n, int(np.ceil((n + 1) * (1 - alpha))))
    thr = np.sort(s_cal)[k - 1]
    sets = []
    for p in p_tst:
        members = [c for c in (0, 1) if (1.0 - (p if c == 1 else 1.0 - p)) <= thr + 1e-12]
        sets.append(tuple(members))
    return sets, p_tst

# ------------------------------------------------------------------ gate
def fit_risk_controlled_gate(ad_cal, std_cal, err_cal, target_risk=0.10):
    """Pick (ad_threshold, std_threshold) on calibration data such that the
    answered set's empirical error <= target_risk, maximising coverage.

    err_cal: 0/1 error indicator of the base predictor on calibration set.
    Returns dict of thresholds and achieved cal error/coverage.
    """
    best = None
    for q_ad in np.arange(0.00, 0.51, 0.05):
        for q_std in np.arange(0.00, 0.71, 0.05):
            t_ad = np.quantile(ad_cal, q_ad)
            t_std = np.quantile(std_cal, q_std)
            answer = (ad_cal >= t_ad) & (std_cal <= t_std)
            if answer.sum() < 10:
                continue
            risk = err_cal[answer].mean()
            cov = answer.mean()
            if risk <= target_risk and (best is None or cov > best["coverage"]):
                best = {"t_ad": float(t_ad), "t_std": float(t_std),
                        "q_ad": float(q_ad), "q_std": float(q_std),
                        "coverage": float(cov), "risk": float(risk)}
    if best is None:  # fall back to most conservative gate
        t_ad, t_std = np.quantile(ad_cal, 0.5), 0.0
        answer = (ad_cal >= t_ad) & (std_cal <= t_std)
        best = {"t_ad": float(t_ad), "t_std": float(t_std),
                "q_ad": 0.5, "q_std": 0.0,
                "coverage": float(answer.mean()),
                "risk": float(err_cal[answer].mean()) if answer.sum() else 1.0}
    return best

def route(ad_max, std, conf_set, gate, t_abstain=0.35):
    """Trust-gated routing for one molecule.

    ANSWER    : in-domain (AD >= t_ad), low disagreement (std <= t_std),
                and conformal set is a singleton.
    ABSTAIN   : far outside domain (max Tanimoto < t_abstain) or empty
                conformal set -> no scientifically defensible answer.
    ESCALATE  : everything in between -> physics-based tools / human expert.
    """
    if ad_max < t_abstain or len(conf_set) == 0:
        return "ABSTAIN"
    if ad_max >= gate["t_ad"] and std <= gate["t_std"] and len(conf_set) == 1:
        return "ANSWER"
    return "ESCALATE"

# ------------------------------------------------------------------ metrics
def ece(y_true, p1, n_bins=10):
    """Expected calibration error (equal-width bins on predicted prob)."""
    bins = np.clip((p1 * n_bins).astype(int), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.sum() == 0:
            continue
        e += m.mean() * abs(p1[m].mean() - y_true[m].mean())
    return float(e)

def risk_coverage(y_true, pred, unc):
    """Risk-coverage curve sorted by ascending uncertainty.

    Returns (coverage array, risk array, AURC).
    """
    order = np.argsort(unc)
    y, p = y_true[order], pred[order]
    risks, covs = [], []
    n = len(y)
    for k in range(10, n + 1, max(1, n // 100)):
        risks.append((y[:k] != p[:k]).mean())
        covs.append(k / n)
    aurc = float(np.trapezoid(risks, covs))
    return np.array(covs), np.array(risks), aurc

def selective_metrics(y_true, pred, routes):
    """Performance of the trust-gated agent.

    Returns dict: coverage, answered error, fraction of all errors captured
    in the ESCALATE+ABSTAIN routes (error-capture yield), per-route stats.
    """
    y = np.asarray(y_true); p = np.asarray(pred); r = np.asarray(routes)
    ans = r == "ANSWER"
    out = {
        "coverage": float(ans.mean()),
        "answered_error": float((y[ans] != p[ans]).mean()) if ans.sum() else None,
        "answered_acc": float((y[ans] == p[ans]).mean()) if ans.sum() else None,
        "n_answer": int(ans.sum()),
        "n_escalate": int((r == "ESCALATE").sum()),
        "n_abstain": int((r == "ABSTAIN").sum()),
    }
    errors = (y != p)
    captured = errors & (r != "ANSWER")
    out["error_capture"] = float(captured.sum() / errors.sum()) if errors.sum() else None
    # accuracy of escalation oracle: if escalated molecules were perfectly
    # resolved by downstream tools, the pipeline error becomes answered_error
    return out
