"""
adagent/agent.py
================
ADAgent: applicability-domain-aware, abstaining toxicity agent.

Pipeline per molecule:
  1. predict with the calibrated ensemble (base predictor)
  2. compute AD score (max Tanimoto to training set)
  3. compute ensemble disagreement + conformal prediction set
  4. trust gate routes: ANSWER | ESCALATE | ABSTAIN
  5. ESCALATE runs the escalation tool registry (physics-based /
     structural-alert tools) and merges evidence into a final,
     cited decision record

The LLM orchestrator is pluggable: set `llm_fn` to any callable
(prompt -> text) such as an OpenAI/Anthropic API wrapper. If none is
provided, a deterministic rule-based reporter is used, so the whole
system runs offline and reproducibly.

Usage:
    python3 agent.py            # demo on sample molecules
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import core

# ------------------------------------------------------------------ tools
class ToolRegistry:
    """Escalation tools a molecule is routed to when the gate says
    ESCALATE. Each tool returns (evidence_text, risk_flag)."""

    def __init__(self):
        self.tools = {
            "structural_alerts": self._structural_alerts,
            "rdkit_descriptors": self._rdkit_descriptors,
            "lipinski_admet": self._lipinski,
        }

    # -- hERG/cardio structural alert stand-in for physics-based tools ----
    def _structural_alerts(self, smi: str):
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smi)
        alerts = []
        if mol is None:
            return "invalid SMILES", True
        smarts = {
            "aromatic_nitro": "[c][N+](=O)[O-]",      # mutagenicity alert
            "aniline_NH2": "[c]N",                    # Ames alert
            "para_dihalobenzene": "c([F,Cl,Br])cc([F,Cl,Br])",
            "thiol": "[SH]",
        }
        for name, patt in smarts.items():
            if Chem.MolFromSmarts(patt) and mol.HasSubstructMatch(Chem.MolFromSmarts(patt)):
                alerts.append(name)
        ev = f"structural alerts: {alerts if alerts else 'none'}"
        return ev, len(alerts) > 0

    def _rdkit_descriptors(self, smi: str):
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Crippen
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return "invalid SMILES", True
        mw, logp, tpsa = (Descriptors.MolWt(mol), Crippen.MolLogP(mol),
                          Descriptors.TPSA(mol))
        ev = (f"MW={mw:.0f} Da, cLogP={logp:.2f}, TPSA={tpsa:.0f} A^2 "
              f"(hERG liability increases with LogP>3 & TPSA<75)")
        return ev, logp > 3 and tpsa < 75

    def _lipinski(self, smi: str):
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return "invalid SMILES", True
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        viol = sum([Descriptors.MolWt(mol) > 500, Crippen.MolLogP(mol) > 5,
                    hbd > 5, hba > 10])
        return f"rule-of-5 violations: {viol}/4", viol >= 2

    def run_all(self, smi: str):
        out = {}
        for name, fn in self.tools.items():
            out[name] = fn(smi)
        return out

# ------------------------------------------------------------------ agent
class ADAgent:
    def __init__(self, X_train, y_train, train_smiles, llm_fn=None,
                 target_risk=0.10, alpha=0.10):
        self.X_train = X_train
        self.models = core.train_ensemble(X_train, y_train)
        self.train_smiles = train_smiles
        self.tools = ToolRegistry()
        self.llm_fn = llm_fn or self._deterministic_reporter
        self.target_risk, self.alpha = target_risk, alpha
        # gate + conformal need a calibration set; split off 15% of train
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(X_train))
        ncal = max(50, int(0.15 * len(X_train)))
        self.cal_idx, self.fit_idx = idx[:ncal], idx[ncal:]
        self.models = core.train_ensemble(X_train[self.fit_idx],
                                          y_train[self.fit_idx])
        self.p_cal, self.std_cal = core.ensemble_predict(self.models, X_train[self.cal_idx])
        self.y_cal = y_train[self.cal_idx]
        self.ad_cal = core.ad_scores(X_train[self.fit_idx],
                                     X_train[self.cal_idx])["max"]
        err_cal = ((self.p_cal > 0.5).astype(int) != self.y_cal).astype(int)
        self.gate = core.fit_risk_controlled_gate(self.ad_cal, self.std_cal,
                                                  err_cal, target_risk)

    # -- deterministic reporter (fallback when no LLM is plugged in) ------
    def _deterministic_reporter(self, prompt: str) -> str:
        return ("Deterministic evidence summary (plug in an LLM via "
                "llm_fn for natural-language synthesis):\n" + prompt)

    # -- main entry ---------------------------------------------------------
    def predict(self, smi: str) -> dict:
        X, valid = core.featurize([smi])
        if len(valid) == 0:
            return {"smiles": smi, "route": "ABSTAIN",
                    "reason": "invalid or empty molecule"}
        x = X[0]
        p1, std = core.ensemble_predict(self.models, x.reshape(1, -1))
        p1, std = float(p1[0]), float(std[0])
        ad = core.ad_scores(self.X_train[self.fit_idx], x.reshape(1, -1))["max"][0]
        conf_sets, _ = core.conformal_sets(self.models,
                                           self.X_train[self.cal_idx],
                                           self.y_cal, x.reshape(1, -1),
                                           self.alpha)
        conf_set = conf_sets[0]
        route = core.route(ad, std, conf_set, self.gate)

        record = {"smiles": smi, "p_toxic": p1, "ensemble_std": std,
                  "ad_max_similarity": float(ad), "conformal_set": conf_set,
                  "route": route, "evidence": []}
        if route == "ANSWER":
            record["prediction"] = int(p1 > 0.5)
            record["evidence"].append(
                f"in-domain (AD={ad:.2f}), low disagreement (std={std:.3f}), "
                f"conformal set {conf_set} @ {1-self.alpha:.0%} coverage")
        elif route == "ESCALATE":
            tool_out = self.tools.run_all(smi)
            for tname, (ev, flag) in tool_out.items():
                record["evidence"].append(f"[{tname}] {ev}"
                                          + ("  <-- RISK FLAG" if flag else ""))
            record["prediction"] = int(p1 > 0.5)
            record["escalated"] = True
        else:
            record["evidence"].append(
                f"outside applicability domain (AD={ad:.2f}) or empty "
                f"conformal set {conf_set}: no defensible answer")
        # natural-language synthesis layer (LLM if provided)
        record["report"] = self.llm_fn(
            f"molecule {smi}; p_toxic={p1:.2f}; AD={ad:.2f}; std={std:.3f}; "
            f"conformal={conf_set}; route={route}; evidence={record['evidence']}")
        return record

# ------------------------------------------------------------------ demo
if __name__ == "__main__":
    df = pd.read_csv(Path(__file__).resolve().parent / "data" / "herg.csv")
    smiles = df["Drug"].tolist()
    X, valid = core.featurize(smiles)
    y = df["Y"].to_numpy()[valid]
    agent = ADAgent(X, y, [smiles[i] for i in valid])

    # probes: a training-set molecule (in-domain -> ANSWER), a drug-like
    # structure outside the training domain, and a metallic salt (far OOD)
    in_domain = smiles[valid[0]]
    probes = [in_domain,
              "CC(=O)Oc1ccccc1C(=O)O",            # aspirin
              "C1CC1(C)C(C)(C)NCCc1ccc(O)c(O)c1",  # a catechol amine
              "[Na+].[Cl-]"]                       # salt: out of domain
    for smi in probes:
        r = agent.predict(smi)
        print(f"\n=== {smi} ===")
        for k in ("route", "p_toxic", "ad_max_similarity", "conformal_set"):
            print(f"  {k}: {r.get(k)}")
        for e in r["evidence"]:
            print(f"  - {e}")
