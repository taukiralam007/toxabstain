"""
adagent/audit.py
================
Leakage audit: for each dataset and split strategy, report the fraction of
test compounds whose nearest training neighbour has Tanimoto similarity
> 0.6 (analog leakage), reproducing the ToxBench (BMC Bioinformatics 2026)
protocol on our datasets.
"""
from pathlib import Path
import numpy as np
import pandas as pd

import core

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "results"

def load(name):
    df = pd.read_csv(f"{DATA}/{name}.csv")
    smiles = df["Drug"].tolist()
    X, valid = core.featurize(smiles)
    return df, smiles, X, valid

def audit(name):
    df, smiles, X, valid = load(name)
    core.set_smiles_for_splits(smiles)  # always the FULL list: split indices are original-frame
    pos_of = {orig: pos for pos, orig in enumerate(valid)}
    rows = []
    for method in ["random", "scaffold", "maxmin"]:
        for seed in [0]:
            tr, va, te = core.split_indices(len(smiles), smiles, method, seed=seed)
            # map original df positions -> fingerprint matrix positions
            trv = np.array([pos_of[i] for i in tr if i in pos_of])
            tev = np.array([pos_of[i] for i in te if i in pos_of])
            ad = core.ad_scores(X[trv], X[tev])
            rows.append({
                "dataset": name, "split": method, "seed": seed,
                "n_train": len(trv), "n_test": len(tev),
                "frac_test_nn_sim_gt0.6": float(ad["frac_gt06"].mean()),
                "median_max_sim": float(np.median(ad["max"])),
                "p10_max_sim": float(np.quantile(ad["max"], 0.10)),
            })
    return rows

if __name__ == "__main__":
    all_rows = []
    for name in ["tox21", "herg", "dili", "ames"]:
        all_rows += audit(name)
        print(f"audited {name}")
    res = pd.DataFrame(all_rows)
    OUT.mkdir(exist_ok=True)
    res.to_csv(f"{OUT}/leakage_audit.csv", index=False)
    print(res.to_string(index=False))
