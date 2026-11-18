"""
adagent/run_experiments.py
==========================
Full evaluation of the AD-aware, abstaining toxicity agent ("ADAgent").

For each dataset x split x task:
  1. train a bagged RF ensemble (ECFP4 features)
  2. compute AD scores (max Tanimoto to train) on validation + test
  3. fit a risk-controlled gate on validation (AD + disagreement
     thresholds s.t. answered-set error <= target while max coverage)
  4. calibrate inductive conformal prediction sets (alpha = 0.1)
  5. route every test molecule: ANSWER / ESCALATE / ABSTAIN
  6. report: AUROC, AUPR, ECE, AURC, and selective-prediction metrics

Outputs: results/main_results.csv, results/per_molecule/ (routes + probs)
"""
import json, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd

import core
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "results"
os.makedirs(f"{OUT}/per_molecule", exist_ok=True)

TARGET_RISK = 0.10   # answered-set error budget
ALPHA = 0.10         # conformal miscoverage

def tasks_of(name, df):
    if name == "tox21":
        return [c for c in df.columns if c not in ("Drug_ID", "Drug")]
    return ["Y"]

def run():
    rows = []
    for name in ["tox21", "herg", "dili", "ames"]:
        df = pd.read_csv(f"{DATA}/{name}.csv")
        smiles = df["Drug"].tolist()
        X, valid = core.featurize(smiles)
        core.set_smiles_for_splits(smiles)
        pos_of = {o: p for p, o in enumerate(valid)}
        tasks = tasks_of(name, df)

        for split in ["random", "scaffold", "maxmin"]:
            tr, va, te = core.split_indices(len(smiles), smiles, split, seed=0)
            trp = np.array([pos_of[i] for i in tr if i in pos_of])
            vap = np.array([pos_of[i] for i in va if i in pos_of])
            tep = np.array([pos_of[i] for i in te if i in pos_of])
            ad_cal = core.ad_scores(X[trp], X[vap])["max"]
            ad_tst = core.ad_scores(X[trp], X[tep])
            leak = float(ad_tst["frac_gt06"].mean())

            for task in tasks:
                y_all = pd.to_numeric(df[task], errors="coerce").to_numpy()
                # per-task labelled subsets
                trl = np.array([p for p in trp if not np.isnan(y_all[valid[p]])])
                val = np.array([p for p in vap if not np.isnan(y_all[valid[p]])])
                tel = np.array([p for p in tep if not np.isnan(y_all[valid[p]])])
                if len(trl) < 50 or len(tel) < 20:
                    continue
                ytr = y_all[valid[trl]].astype(int)
                yva = y_all[valid[val]].astype(int)
                yte = y_all[valid[tel]].astype(int)
                if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                    continue

                models = core.train_ensemble(X[trl], ytr, n_models=3, n_trees=60, seed=0)
                p_cal, std_cal = core.ensemble_predict(models, X[val])
                p_tst, std_tst = core.ensemble_predict(models, X[tel])
                pred = (p_tst > 0.5).astype(int)

                # risk-controlled gate fitted on validation
                # (AD scores filtered to the per-task labelled subset)
                val_mask = np.isin(vap, val)
                ad_cal_t = ad_cal[val_mask]
                err_cal = ((p_cal > 0.5).astype(int) != yva).astype(int)
                gate = core.fit_risk_controlled_gate(ad_cal_t, std_cal, err_cal, TARGET_RISK)

                # conformal sets
                conf_sets, _ = core.conformal_sets(models, X[val], yva, X[tel], ALPHA)

                # AD scores for the per-task labelled test subset
                tel_mask = np.isin(tep, tel)
                ad_max_t = ad_tst["max"][tel_mask]

                # route every test molecule
                routes = np.array([
                    core.route(ad_max_t[k], std_tst[k], conf_sets[k], gate)
                    for k in range(len(tel))])

                sel = core.selective_metrics(yte, pred, routes)
                try:
                    auroc = roc_auc_score(yte, p_tst)
                    aupr = average_precision_score(yte, p_tst)
                except ValueError:
                    auroc = aupr = float("nan")
                ece_v = core.ece(yte, p_tst)
                _, _, aurc = core.risk_coverage(yte, pred, std_tst)

                base_err = float((pred != yte).mean())
                rows.append({
                    "dataset": name, "split": split, "task": task,
                    "n_train": len(trl), "n_test": len(tel),
                    "frac_pos": float(yte.mean()),
                    "leakage_frac_gt06": leak,
                    "AUROC": auroc, "AUPR": aupr, "ECE": ece_v, "AURC": aurc,
                    "base_error": base_err,
                    **{f"gate_{k}": v for k, v in gate.items()},
                    **sel,
                })

                # per-molecule dump
                pd.DataFrame({
                    "smiles": [smiles[valid[p]] for p in tel],
                    "y": yte, "pred": pred, "p1": p_tst, "std": std_tst,
                    "ad_max": ad_max_t, "route": routes,
                }).to_csv(f"{OUT}/per_molecule/{name}_{split}_{task}.csv", index=False)

                print(f"{name:6s} {split:8s} {task:16s} "
                      f"AUROC={auroc:.3f} base_err={base_err:.3f} "
                      f"cov={sel['coverage']:.2f} ans_err={sel['answered_error'] if sel['answered_error'] is not None else float('nan'):.3f} "
                      f"capture={sel['error_capture'] if sel['error_capture'] is not None else float('nan'):.2f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/main_results.csv", index=False)
    print("\nsaved", f"{OUT}/main_results.csv", "| rows:", len(res))

if __name__ == "__main__":
    run()
