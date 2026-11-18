# ADAgent — Applicability-Domain-Aware, Abstaining Toxicity Agent

A trust-gated toxicity prediction system: for every novel compound it computes
an applicability-domain (AD) score, a calibrated uncertainty estimate, and a
conformal prediction set, then routes the molecule to one of three actions:

| Route | Meaning | Trigger |
|---|---|---|
| **ANSWER** | confident prediction | in-domain (AD >= gate threshold), low ensemble disagreement, singleton conformal set |
| **ESCALATE** | run physics/chemistry tools, then human review | borderline domain or ambiguous uncertainty |
| **ABSTAIN** | no scientifically defensible answer | far out of domain or empty conformal set |

The gate thresholds are fitted on a held-out calibration split under a
**risk-controlled** criterion: answered-set error <= target (default 10%)
while maximizing coverage.

## Layout

```
data_prep.py       # download Tox21/hERG/DILI/Ames from TDC
core.py            # ECFP4 features, splits, AD, RF-ensemble UQ, conformal, gate, metrics
audit.py           # leakage audit (analog leakage by split strategy)
run_experiments.py # full benchmark: 45 dataset x split x task evaluations
analyze.py         # summary tables + publication figures
agent.py           # pluggable LLM orchestrator + escalation tool registry
data/              # downloaded datasets (CSV)
results/           # main_results.csv, summaries, per-molecule route dumps
figs/              # fig1 leakage, fig2 OOD + gate, fig3 risk-coverage
```

## Reproduce

```bash
pip install scikit-learn pandas numpy matplotlib rdkit PyTDC
python3 data_prep.py        # ~2 min
python3 audit.py            # leakage audit
python3 run_experiments.py  # ~10 min on 2 cores
python3 analyze.py          # tables + figures
python3 agent.py            # route demo (ANSWER / ESCALATE / ABSTAIN)
```

## Key design choices

- **ECFP4 (Morgan r=2, 2048-bit)** fingerprints; bagged RandomForest ensembles
  (3 x 60 trees, class-balanced) — deliberately simple base predictors so the
  *gating*, not the predictor, is the object of study.
- **AD score**: max Tanimoto similarity to the training set, following the
  leakage-audit protocol of ToxBench (BMC Bioinformatics 2026).
- **Splits**: random, Bemis-Murcko scaffold, and a novel **cluster-closure
  maxmin** split that moves whole >0.6-similarity components into the test set,
  guaranteeing zero analog leakage by construction.
- **Conformal prediction** (inductive, alpha = 0.1) gives distribution-free
  marginal coverage; ambiguous ({0,1}) or empty sets force abstention.
- **Risk-controlled gate**: grid search over (AD, disagreement) quantiles on
  the calibration split, subject to the answered-set error budget.

## Plugging in an LLM

`ADAgent(llm_fn=...)` accepts any callable `prompt -> text` (e.g., an
OpenAI/Anthropic API wrapper). Without one, a deterministic reporter is used,
so results are fully reproducible offline. The escalation tool registry
(`ToolRegistry`) is where docking/FEP/ADMET services plug in.
