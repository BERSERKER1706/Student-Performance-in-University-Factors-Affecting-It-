"""Part B analysis: does standard practice recover the effects planted in the simulation?"""
import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline

from . import plotting as P
from .config import FIG, TAB, SEED, TRUE, BRANCHES, BRANCH_OFFSET, STUDY_BOOST
from .simulate import CONTINUOUS

LABEL = {"class12_pct": "Class 12 %", "attendance": "Attendance %", "study_hours": "Study hours / day",
         "sleep_hours": "Sleep hours", "screen_hours": "Recreational screen h", "stress": "Stress (0-10)",
         "paid_work": "Part-time paid work", "mentor_access": "Has a mentor", "first_gen": "First-generation",
         "hostel": "Lives in hostel", "tier_2": "Tier 2 vs Tier 1", "tier_3": "Tier 3 vs Tier 1"}


def run(df):
    P.apply(); res = {"n": int(len(df))}
    d = df.copy()
    d["tier"] = d["tier"].str.replace(" ", "")
    for c in CONTINUOUS:                               # standardise on the full simulated sample
        d[c + "_z"] = (d[c] - d[c].mean()) / d[c].std()
    cc = d.dropna(subset=[c + "_z" for c in CONTINUOUS]).copy()
    res["n_complete"] = int(len(cc))

    # 1. branch comparison (raw)
    order = d.groupby("branch").cgpa.mean().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, b in enumerate(order):
        v = d.cgpa[d.branch == b]
        ax.scatter(np.random.default_rng(i).normal(i, 0.08, len(v)), v, s=4, alpha=0.18, color=P.CAT[0], linewidths=0)
        ax.hlines(v.mean(), i - 0.3, i + 0.3, color=P.INK, lw=2)
        ax.text(i, 10.15, f"{v.mean():.2f}", ha="center", fontsize=9, color=P.INK2)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([f"{b}\n(n={int((d.branch == b).sum())})" for b in order])
    ax.set_ylim(4, 10.5); ax.set_ylabel("CGPA (10-point)"); ax.grid(axis="x", visible=False)
    ax.set_title("Simulated cohort: CGPA by branch (bar = mean)")
    P.save(fig, FIG / "B1_branch_cgpa.png")
    res["branch_means"] = d.groupby("branch").cgpa.mean().round(3).to_dict()

    # 2. mixed model with institution random intercept
    rhs = " + ".join([c + "_z" for c in CONTINUOUS] + ["paid_work", "mentor_access", "first_gen", "hostel", "C(tier, Treatment('Tier1'))", "C(branch, Treatment('EE'))"])
    mm = smf.mixedlm(f"cgpa ~ {rhs}", cc, groups=cc["institution"]).fit(reml=True)
    ols = smf.ols(f"cgpa ~ {rhs}", cc).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(cc.institution)[0]})
    ci = mm.conf_int()
    rows = []
    def add(name, term, true):
        rows.append((name, true, mm.params[term], ci.loc[term, 0], ci.loc[term, 1]))
    for c in CONTINUOUS: add(LABEL[c], c + "_z", TRUE[c])
    for c in ["paid_work", "mentor_access", "first_gen", "hostel"]: add(LABEL[c], c, TRUE[c])
    add(LABEL["tier_2"], "C(tier, Treatment('Tier1'))[T.Tier2]", TRUE["tier_2"])
    add(LABEL["tier_3"], "C(tier, Treatment('Tier1'))[T.Tier3]", TRUE["tier_3"])
    for b in BRANCHES:
        if b != "EE":
            add(f"Branch {b} vs EE", f"C(branch, Treatment('EE'))[T.{b}]", BRANCH_OFFSET[b])
    rec = pd.DataFrame(rows, columns=["effect", "true", "estimate", "ci_lo", "ci_hi"])
    rec["covered"] = (rec.true >= rec.ci_lo) & (rec.true <= rec.ci_hi)
    rec["error"] = rec.estimate - rec.true
    rec.to_csv(TAB / "B_recovered_vs_true.csv", index=False)
    icc = float(mm.cov_re.iloc[0, 0] / (mm.cov_re.iloc[0, 0] + mm.scale))
    res["mixed"] = {"icc_institution": icc, "coverage": float(rec.covered.mean()), "n_effects": int(len(rec)),
                    "residual_sd": float(np.sqrt(mm.scale)), "true_noise_sd": TRUE["noise_sd"]}
    s = rec.iloc[::-1].reset_index(drop=True); y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(7.8, 7))
    ax.errorbar(s.estimate, y, xerr=[s.estimate - s.ci_lo, s.ci_hi - s.estimate], fmt="o", ms=5, color=P.CAT[0], ecolor=P.CAT[0], lw=1.2, mec=P.SURFACE, label="Estimate (95% CI)")
    ax.scatter(s["true"], y, marker="D", s=26, color=P.CAT[1], zorder=3, label="True planted value")
    ax.axvline(0, color=P.INK2, lw=0.8); ax.set_yticks(y); ax.set_yticklabels(s.effect)
    ax.set_xlabel("Effect on CGPA (per +1 SD for continuous, vs reference for categories)")
    ax.set_title(f"Recovery check: {int(rec.covered.sum())} of {len(rec)} true effects inside the 95% CI")
    ax.legend(loc="lower right"); ax.grid(axis="y", visible=False)
    P.save(fig, FIG / "B2_recovered_vs_true.png")

    # 3. does study time pay off equally across branches?
    rows = []
    ctrl = " + ".join([c + "_z" for c in CONTINUOUS if c != "study_hours"] + ["paid_work", "mentor_access", "first_gen", "hostel", "C(tier)"])
    for b in BRANCHES:
        sub = cc[cc.branch == b]
        f = smf.ols(f"cgpa ~ study_hours_z + {ctrl}", sub).fit(cov_type="HC3")
        lo, hi = f.conf_int().loc["study_hours_z"]
        rows.append((b, f.params["study_hours_z"], lo, hi, TRUE["study_hours"] + STUDY_BOOST[b], len(sub)))
    sl = pd.DataFrame(rows, columns=["branch", "slope", "ci_lo", "ci_hi", "true", "n"]).sort_values("slope")
    sl.to_csv(TAB / "B_study_slope_by_branch.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4)); y = np.arange(len(sl))
    ax.errorbar(sl.slope, y, xerr=[sl.slope - sl.ci_lo, sl.ci_hi - sl.slope], fmt="o", ms=6, color=P.CAT[0], ecolor=P.CAT[0], lw=1.3, mec=P.SURFACE, label="Estimate (95% CI)")
    ax.scatter(sl["true"], y, marker="D", s=28, color=P.CAT[1], zorder=3, label="True planted value")
    ax.set_yticks(y); ax.set_yticklabels([f"{b} (n={n})" for b, n in zip(sl.branch, sl.n)])
    ax.set_xlabel("CGPA gain per +1 SD of study time"); ax.set_title("Return on study time by branch")
    ax.legend(loc="lower right"); ax.grid(axis="y", visible=False)
    P.save(fig, FIG / "B3_study_by_branch.png")
    res["study_slopes"] = sl.set_index("branch")[["slope", "ci_lo", "ci_hi", "true"]].round(3).to_dict("index")
    hetero = smf.ols(f"cgpa ~ study_hours_z*C(branch, Treatment('EE')) + {ctrl}", cc).fit()
    base = smf.ols(f"cgpa ~ study_hours_z + C(branch, Treatment('EE')) + {ctrl}", cc).fit()
    from statsmodels.stats.anova import anova_lm
    res["interaction_F_p"] = float(anova_lm(base, hetero).iloc[1]["Pr(>F)"])

    # 4. predictive model + permutation importance
    feats = CONTINUOUS + ["paid_work", "mentor_access", "first_gen", "hostel"]
    X = pd.concat([d[feats], pd.get_dummies(d[["branch", "tier"]], drop_first=False).astype(int)], axis=1)
    mk = lambda: make_pipeline(SimpleImputer(strategy="median"), GradientBoostingRegressor(random_state=SEED, n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.8))
    r2 = cross_val_score(mk(), X, d.cgpa, cv=KFold(5, shuffle=True, random_state=SEED), scoring="r2")
    res["gbm_r2"] = {"mean": float(r2.mean()), "sd": float(r2.std())}
    res["ols_r2_cluster_model"] = float(ols.rsquared)
    Xtr, Xte, ytr, yte = train_test_split(X, d.cgpa, test_size=0.25, random_state=SEED)
    mdl = mk().fit(Xtr, ytr)
    pi = permutation_importance(mdl, Xte, yte, n_repeats=30, random_state=SEED, scoring="r2")
    imp = pd.Series(pi.importances_mean, index=X.columns)
    grouped = {LABEL.get(c, c): imp[c] for c in feats}
    grouped["Branch (all)"] = imp[[c for c in X if c.startswith("branch_")]].sum()
    grouped["Institution tier (all)"] = imp[[c for c in X if c.startswith("tier_")]].sum()
    g = pd.Series(grouped).sort_values()
    g.rename("importance").to_csv(TAB / "B_permutation_importance.csv")
    fig, ax = plt.subplots(figsize=(7.5, 5)); y = np.arange(len(g))
    ax.barh(y, g.values, color=P.CAT[0], height=0.55); ax.axvline(0, color=P.INK2, lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(g.index); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Drop in held-out R² when shuffled"); ax.set_title(f"What predicts CGPA in the simulated cohort (R² ≈ {r2.mean():.2f})")
    P.save(fig, FIG / "B4_importance.png")
    res["top_importance"] = g.sort_values(ascending=False).head(5).round(3).to_dict()
    json.dump(res, open(TAB / "results_india.json", "w"), indent=2, default=float)
    return res
