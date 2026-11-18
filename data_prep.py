"""Download toxicity datasets from Therapeutics Data Commons (TDC).

Endpoint set:
  - tox21 : 12 in-vitro NR/stress-response assays (multi-task)
  - herg  : hERG channel blockade (cardiotoxicity)
  - dili  : drug-induced liver injury
  - ames  : bacterial mutagenicity
(ClinTox excluded: 1,379 molecules / extreme imbalance -> seed variance
 +-0.085-0.160 AUROC per ToxBench BMC Bioinformatics 2026.)
"""
from pathlib import Path
from tdc.single_pred import Tox

DATA = Path(__file__).resolve().parent / "data"
DATA.mkdir(exist_ok=True)

def save(df, name):
    df.to_csv(DATA / f"{name}.csv", index=False)
    print(f"{name}: {len(df)} rows | cols: {list(df.columns)[:16]}")

# Tox21 is stored with a label_name per task; fetch all 12 and merge on molecule
from tdc.utils import retrieve_label_name_list
labels = retrieve_label_name_list("tox21")
frames = [Tox(name="tox21", label_name=lab).get_data()[["Drug_ID", "Drug", "Y"]].rename(columns={"Y": lab}) for lab in labels]
import functools, pandas as pd
tox21 = functools.reduce(lambda a, b: pd.merge(a, b, on=["Drug_ID", "Drug"], how="outer"), frames)
save(tox21, "tox21")

for name in ["herg", "dili", "ames"]:
    save(Tox(name=name).get_data(), name)

print("\nDone. Tox21 tasks:", labels)
