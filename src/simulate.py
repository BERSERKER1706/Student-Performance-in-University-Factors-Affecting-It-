"""Part B: SIMULATED Indian multi-branch cohort (10-point CGPA). Not real students.

Every effect that drives CGPA is declared in config.TRUE, so analysis_india.py can check
whether standard methods recover what was planted. Nothing here is a finding about real students.
"""
import numpy as np
import pandas as pd
from .config import (BRANCHES, BRANCH_SHARE, BRANCH_OFFSET, N_INSTITUTIONS, N_STUDENTS, TRUE,
                     STUDY_BOOST, SEED, PROCESSED)

CONTINUOUS = ["class12_pct", "attendance", "study_hours", "sleep_hours", "screen_hours", "stress"]


def _z(x):
    return (x - x.mean()) / x.std()


def simulate(n=N_STUDENTS, seed=SEED):
    rng = np.random.default_rng(seed)
    inst = pd.DataFrame({
        "institution": [f"INST{i + 1:02d}" for i in range(N_INSTITUTIONS)],
        "tier": ["Tier 1"] * 4 + ["Tier 2"] * 4 + ["Tier 3"] * 4,
        "inst_effect": rng.normal(0, 0.15, N_INSTITUTIONS),
    })
    d = pd.DataFrame({"institution": rng.choice(inst.institution, n)}).merge(inst, on="institution")
    d["branch"] = rng.choice(BRANCHES, len(d), p=BRANCH_SHARE)
    d["gender"] = rng.choice(["Male", "Female"], len(d), p=[0.62, 0.38])
    d["first_gen"] = (rng.random(len(d)) < 0.35).astype(int)
    d["hostel"] = (rng.random(len(d)) < 0.45).astype(int)
    d["mentor_access"] = (rng.random(len(d)) < np.where(d.tier == "Tier 1", 0.55, 0.35)).astype(int)
    d["paid_work"] = (rng.random(len(d)) < np.where(d.first_gen == 1, 0.22, 0.10)).astype(int)

    tier_shift = d.tier.map({"Tier 1": 5, "Tier 2": 0, "Tier 3": -5})
    d["class12_pct"] = np.clip(rng.normal(77, 10, len(d)) + tier_shift, 45, 99).round(1)
    d["attendance"] = np.clip(rng.normal(76, 11, len(d)) + 0.12 * (d.class12_pct - 77) - 3 * d.paid_work, 40, 100).round(1)
    d["study_hours"] = np.clip(rng.normal(3.0, 1.3, len(d)) + 0.02 * (d.class12_pct - 77), 0, 10).round(1)
    d["sleep_hours"] = np.clip(rng.normal(6.7, 1.0, len(d)) + 0.3 * d.hostel * -1, 3.5, 10).round(1)
    d["screen_hours"] = np.clip(rng.normal(3.8, 1.6, len(d)) - 0.15 * d.study_hours, 0.2, 12).round(1)
    d["stress"] = np.clip(rng.normal(5.0, 1.8, len(d)) + 0.25 * d.paid_work * 2 - 0.2 * (d.sleep_hours - 6.7), 0, 10).round(1)

    z = {c: _z(d[c]) for c in CONTINUOUS}
    study_extra = d.branch.map(STUDY_BOOST) * z["study_hours"]
    cg = (7.0 + d.branch.map(BRANCH_OFFSET) + d.inst_effect
          + TRUE["class12_pct"] * z["class12_pct"] + TRUE["attendance"] * z["attendance"]
          + TRUE["study_hours"] * z["study_hours"] + study_extra
          + TRUE["sleep_hours"] * z["sleep_hours"] + TRUE["screen_hours"] * z["screen_hours"]
          + TRUE["stress"] * z["stress"] + TRUE["paid_work"] * d.paid_work
          + TRUE["mentor_access"] * d.mentor_access + TRUE["first_gen"] * d.first_gen
          + TRUE["hostel"] * d.hostel
          + d.tier.map({"Tier 1": 0, "Tier 2": TRUE["tier_2"], "Tier 3": TRUE["tier_3"]})
          + rng.normal(0, TRUE["noise_sd"], len(d)))
    d["cgpa"] = np.clip(cg, 4.0, 10.0).round(2)
    # a little missing data in self-reported variables
    for c in ["sleep_hours", "screen_hours", "stress"]:
        d.loc[rng.random(len(d)) < 0.03, c] = np.nan
    d.insert(0, "student_id", [f"S{i + 1:05d}" for i in range(len(d))])
    d = d.drop(columns=["inst_effect"])
    PROCESSED.mkdir(parents=True, exist_ok=True)
    d.to_csv(PROCESSED / "simulated_india_multibranch.csv", index=False)
    return d
