"""
adagent/analyze.py
==================
Aggregate results and produce publication-style figures:
  fig1: analog leakage by dataset x split (random vs scaffold vs maxmin)
  fig2: mean AUROC by split (OOD degradation) + answered-error vs coverage
  fig3: risk-coverage curves (mean over Tox21 tasks) per split
  summary tables -> results/summary_*.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"
F = ROOT / "figs"
F.mkdir(exist_ok=True)
res = pd.read_csv(f"{R}/main_results.csv")
leak = pd.read_csv(f"{R}/leakage_audit.csv")

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})

# ---------------- Fig 1: leakage audit
fig, ax = plt.subplots(figsize=(6.2, 3.2))
datasets = ["tox21", "ames", "herg", "dili"]
labels = ["Tox21", "Ames", "hERG", "DILI"]
splits = ["random", "scaffold", "maxmin"]
colors = {"random": "#c0392b", "scaffold": "#e67e22", "maxmin": "#27ae60"}
w = 0.26
x = np.arange(len(datasets))
for i, s in enumerate(splits):
    vals = [leak[(leak.dataset == d) & (leak.split == s)]["frac_test_nn_sim_gt0.6"].mean() * 100
            for d in datasets]
    bars = ax.bar(x + (i - 1) * w, vals, w, label=s, color=colors[s])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("test molecules with a >0.6 train\nnearest-neighbour (analog leakage, %)")
ax.set_title("Leakage audit: analog leakage by split strategy")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{F}/fig1_leakage.png"); plt.close(fig)

# ---------------- Fig 2: AUROC by split + gate trade-off
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
# (a) mean AUROC per dataset x split
ax = axes[0]
for j, d in enumerate(datasets):
    vals = [res[(res.dataset == d) & (res.split == s)]["AUROC"].mean() for s in splits]
    ax.plot(range(3), vals, "o-", label=labels[j])
ax.set_xticks(range(3)); ax.set_xticklabels(["random", "scaffold", "maxmin"])
ax.set_ylabel("mean AUROC"); ax.set_ylim(0.6, 0.95)
ax.set_title("(a) Discrimination under harder splits")
ax.legend(frameon=False, fontsize=8)
# (b) base error vs answered error (all rows with coverage>0)
ax = axes[1]
sub = res[(res.coverage > 0) & res.answered_error.notna()]
for s, c in colors.items():
    ss = sub[sub.split == s]
    ax.scatter(ss.base_error, ss.answered_error, s=22, alpha=0.75, color=c, label=s)
lim = [0, max(sub.base_error.max(), 0.36)]
ax.plot(lim, lim, "k--", lw=0.8, label="y = x (no gain)")
ax.set_xlabel("error if every molecule is answered")
ax.set_ylabel("error on ANSWERED set (gated)")
ax.set_title("(b) Trust gate: answered-set error vs baseline")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(f"{F}/fig2_ood_gate.png"); plt.close(fig)

# ---------------- Fig 3: risk-coverage curves (Tox21, mean over tasks)
import core  # for risk_coverage on per-molecule dumps
fig, ax = plt.subplots(figsize=(4.6, 3.4))
for s, c in colors.items():
    curves = []
    for task in [c2 for c2 in ["NR-AR", "NR-AhR", "SR-ARE", "SR-p53", "NR-ER-LBD", "SR-MMP"]]:
        try:
            pm = pd.read_csv(f"{R}/per_molecule/tox21_{s}_{task}.csv")
        except FileNotFoundError:
            continue
        cov, risk, _ = core.risk_coverage(pm.y.to_numpy(), pm.pred.to_numpy(), pm["std"].to_numpy())
        curves.append((cov, risk))
    if curves:
        n = min(len(c[0]) for c in curves)
        mean_risk = np.mean([c[1][:n] for c in curves], axis=0)
        mean_cov = curves[0][0][:n]
        ax.plot(mean_cov, mean_risk, color=c, label=f"{s} (Tox21 mean)")
ax.set_xlabel("coverage (fraction of molecules answered)")
ax.set_ylabel("risk (error among answered)")
ax.set_title("Risk-coverage: uncertainty-sorted selective prediction")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(f"{F}/fig3_risk_coverage.png"); plt.close(fig)

# ---------------- Summary tables
def summarize(df, group):
    g = df.groupby(group).agg(
        mean_AUROC=("AUROC", "mean"),
        mean_AUPR=("AUPR", "mean"),
        mean_ECE=("ECE", "mean"),
        mean_base_error=("base_error", "mean"),
        mean_answered_error=("answered_error", "mean"),
        mean_coverage=("coverage", "mean"),
        mean_error_capture=("error_capture", "mean"),
        n=("task", "count"),
    ).round(3)
    return g

by_split = summarize(res, "split")
by_ds = summarize(res, ["dataset", "split"])
by_split.to_csv(f"{R}/summary_by_split.csv")
by_ds.to_csv(f"{R}/summary_by_dataset_split.csv")

# improvement stats
sub = res[(res.coverage > 0) & res.answered_error.notna()]
imp = ((sub.base_error - sub.answered_error) / sub.base_error)
print("=== Summary by split ===")
print(by_split.to_string())
print("\n=== Gate effect (rows with coverage>0) ===")
print(f"mean relative error reduction on answered set: {imp.mean()*100:.1f}%")
print(f"median: {imp.median()*100:.1f}% | mean coverage: {sub.coverage.mean()*100:.1f}%")
print(f"mean error capture (errors routed to ESCALATE/ABSTAIN): {sub.error_capture.mean()*100:.1f}%")
print(f"rows where gate could not meet 10% risk (coverage=0): {(res.coverage==0).sum()} / {len(res)}")
print("\nfigures saved to figs/")
