"""Part A: what is associated with CGPA in the real CSE survey?"""
import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline

from . import plotting as P
from .config import FIG, TAB, SEED, N_BOOT

NUMERIC = ["study_hours", "study_sessions", "social_hours", "skill_hours", "attendance", "log_income", "age"]
BINARY = {  # column -> (label, reference level)
    "gender": "Male", "own_pc": "No", "smartphone": "No", "consultancy": "No", "cocurricular": "No",
    "health_issue": "No", "disability": "No", "uni_transport": "No", "learning_mode": "Offline",
    "living_with": "Family", "relationship": "Single",
}
LABEL = {
    "study_hours": "Study hours / day", "study_sessions": "Study sessions / day", "social_hours": "Social media h/day",
    "skill_hours": "Skill-building h/day", "attendance": "Class attendance %", "log_income": "Family income (log10)",
    "age": "Age", "prev_sgpa": "Previous-semester SGPA", "semester": "Semester",
    "english": "English proficiency", "english_score": "English proficiency", "gender": "Female vs male", "own_pc": "Owns a PC",
    "smartphone": "Uses smartphone", "consultancy": "Attends teacher consultation", "cocurricular": "Co-curricular activity",
    "health_issue": "Health issue", "disability": "Physical disability", "uni_transport": "Uses uni transport",
    "learning_mode": "Prefers online learning", "living_with": "Lives away from family", "relationship": "In a relationship",
    "scholarship": "Merit scholarship", "suspension": "Ever suspended", "probation": "Ever on probation",
}
# Variables that are CONSEQUENCES of performance (or just elapsed time) are kept out of the factor models.
CONSEQUENCES = ["scholarship", "probation", "suspension", "credits"]


def _boot_ci(x, y, fn, rng, n=N_BOOT):
    idx = np.arange(len(x)); out = []
    for _ in range(n):
        b = rng.choice(idx, len(idx)); out.append(fn(x[b], y[b]))
    return np.percentile(out, [2.5, 97.5])


def _design(df):
    d = df.copy()
    d["english_score"] = d["english"].map({"Basic": 0, "Intermediate": 1, "Advance": 2})
    for c, ref in BINARY.items():
        d[c] = (d[c] != ref).astype(int)
    return d


def run(df):
    P.apply(); rng = np.random.default_rng(SEED); res = {}
    d = _design(df)
    res["n"] = int(len(d)); res["cgpa_mean"] = float(d.cgpa.mean()); res["cgpa_sd"] = float(d.cgpa.std())

    # 1. correlations with bootstrap CIs
    rows = []
    for c in NUMERIC + ["prev_sgpa", "semester", "english_score"]:
        s = d[[c, "cgpa"]].dropna()
        rho = stats.spearmanr(s[c], s.cgpa)[0]
        lo, hi = _boot_ci(s[c].to_numpy(), s.cgpa.to_numpy(), lambda a, b: stats.spearmanr(a, b)[0], rng, 500)
        rows.append((c if c != "english_score" else "english", rho, lo, hi, len(s)))
    corr = pd.DataFrame(rows, columns=["factor", "spearman", "ci_lo", "ci_hi", "n"]).sort_values("spearman")
    corr.to_csv(TAB / "A_correlations.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    y = np.arange(len(corr))
    col = [P.POS if r > 0 else P.NEG for r in corr.spearman]
    ax.barh(y, corr.spearman, color=col, height=0.55)
    ax.errorbar(corr.spearman, y, xerr=[corr.spearman - corr.ci_lo, corr.ci_hi - corr.spearman], fmt="none", ecolor=P.INK2, lw=1, capsize=2)
    ax.axvline(0, color=P.INK2, lw=0.8); ax.set_yticks(y); ax.set_yticklabels([LABEL.get(f, f) for f in corr.factor])
    ax.set_xlabel("Spearman correlation with CGPA (95% bootstrap CI)")
    ax.set_title("Prior results dominate; habits barely register"); ax.grid(axis="y", visible=False)
    P.save(fig, FIG / "A1_correlations.png")
    res["corr"] = corr.set_index("factor")["spearman"].round(3).to_dict()

    # 2. group differences in mean CGPA
    g = []
    for c in list(BINARY) + ["scholarship", "probation", "suspension"]:
        flag = d[c] if c in BINARY else (df[c] == "Yes").astype(int).loc[d.index]
        a, b = d.cgpa[flag == 1].to_numpy(), d.cgpa[flag == 0].to_numpy()
        diff = a.mean() - b.mean()
        bs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(1000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        g.append((c, diff, lo, hi, len(a), len(b), stats.mannwhitneyu(a, b)[1]))
    grp = pd.DataFrame(g, columns=["factor", "diff", "ci_lo", "ci_hi", "n_yes", "n_no", "p_mwu"]).sort_values("diff")
    grp.to_csv(TAB / "A_group_differences.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    y = np.arange(len(grp))
    cons = grp.factor.isin(CONSEQUENCES)
    ax.errorbar(grp["diff"], y, xerr=[grp["diff"] - grp.ci_lo, grp.ci_hi - grp["diff"]], fmt="none", ecolor=P.INK2, lw=1, capsize=2, zorder=1)
    ax.scatter(grp["diff"], y, s=48, c=[P.NEG if c else P.POS for c in cons], zorder=2, edgecolor=P.SURFACE, linewidth=1.5)
    ax.axvline(0, color=P.INK2, lw=0.8); ax.set_yticks(y); ax.set_yticklabels([LABEL.get(f, f) for f in grp.factor])
    ax.set_xlabel("Difference in mean CGPA (group with trait minus without), 95% CI")
    ax.set_title("Group gaps: blue = possible factor, orange = result of performance")
    ax.grid(axis="y", visible=False)
    P.save(fig, FIG / "A2_group_gaps.png")

    # 3. OLS with robust SEs, standardised numeric predictors
    use = d.dropna(subset=NUMERIC + ["prev_sgpa", "semester", "cgpa"]).copy()
    for c in NUMERIC + ["prev_sgpa", "semester"]:
        use[c] = (use[c] - use[c].mean()) / use[c].std()
    base = " + ".join(NUMERIC + ["semester", "english_score"] + list(BINARY))
    m1 = smf.ols(f"cgpa ~ {base}", use).fit(cov_type="HC3")
    m2 = smf.ols(f"cgpa ~ prev_sgpa + {base}", use).fit(cov_type="HC3")
    res["ols"] = {"n": int(m1.nobs), "r2_without_prior": float(m1.rsquared), "r2_with_prior": float(m2.rsquared)}
    coef = pd.DataFrame({"m1_coef": m1.params, "m1_lo": m1.conf_int()[0], "m1_hi": m1.conf_int()[1], "m1_p": m1.pvalues,
                         "m2_coef": m2.params, "m2_lo": m2.conf_int()[0], "m2_hi": m2.conf_int()[1], "m2_p": m2.pvalues}).drop("Intercept")
    coef.to_csv(TAB / "A_ols_coefficients.csv")
    show = coef.dropna(subset=["m1_coef"]).sort_values("m1_coef")
    fig, ax = plt.subplots(figsize=(7.5, 6.4)); y = np.arange(len(show))
    for j, (k, lab, colr, off) in enumerate([("m1", "Without prior SGPA", P.CAT[0], -0.15), ("m2", "Controlling for prior SGPA", P.CAT[1], 0.15)]):
        ax.errorbar(show[f"{k}_coef"], y + off, xerr=[show[f"{k}_coef"] - show[f"{k}_lo"], show[f"{k}_hi"] - show[f"{k}_coef"]],
                    fmt="o", ms=4.5, color=colr, ecolor=colr, lw=1, label=lab, mec=P.SURFACE)
    ax.axvline(0, color=P.INK2, lw=0.8); ax.set_yticks(y); ax.set_yticklabels([LABEL.get(f, f) for f in show.index])
    ax.set_xlabel("CGPA change per +1 SD (numeric) or vs reference (yes/no), robust 95% CI")
    ax.set_title("Adjusted associations with CGPA"); ax.legend(loc="upper left"); ax.grid(axis="y", visible=False)
    P.save(fig, FIG / "A3_regression.png")

    # 4. predictive model + grouped permutation importance (held-out)
    feats = NUMERIC + ["semester", "english_score"] + list(BINARY)
    X = d[feats + ["prev_sgpa"]]; yv = d.cgpa
    kf = KFold(5, shuffle=True, random_state=SEED)
    mk = lambda: make_pipeline(SimpleImputer(strategy="median"), GradientBoostingRegressor(random_state=SEED, n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8))
    r2_wo = cross_val_score(mk(), X[feats], yv, cv=kf, scoring="r2")
    r2_w = cross_val_score(mk(), X, yv, cv=kf, scoring="r2")
    base_dummy = 0.0
    res["gbm"] = {"r2_without_prior_mean": float(r2_wo.mean()), "r2_without_prior_sd": float(r2_wo.std()),
                  "r2_with_prior_mean": float(r2_w.mean()), "r2_with_prior_sd": float(r2_w.std())}
    Xtr, Xte, ytr, yte = train_test_split(X, yv, test_size=0.25, random_state=SEED)
    mdl = mk().fit(Xtr[feats], ytr)
    pi = permutation_importance(mdl, Xte[feats], yte, n_repeats=30, random_state=SEED, scoring="r2")
    imp = pd.DataFrame({"factor": feats, "importance": pi.importances_mean, "sd": pi.importances_std}).sort_values("importance")
    imp.to_csv(TAB / "A_permutation_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 5)); y = np.arange(len(imp))
    ax.barh(y, imp.importance, xerr=imp.sd, color=P.CAT[0], height=0.55, error_kw=dict(ecolor=P.INK2, lw=1, capsize=2))
    ax.axvline(0, color=P.INK2, lw=0.8); ax.set_yticks(y); ax.set_yticklabels([LABEL.get(f, f) for f in imp.factor])
    ax.set_xlabel("Drop in held-out R² when the factor is shuffled (no prior-SGPA in model)")
    ax.set_title(f"Which factors help predict CGPA? (model R² ≈ {r2_wo.mean():.2f})"); ax.grid(axis="y", visible=False)
    P.save(fig, FIG / "A4_importance.png")

    # 5. dose-response: attendance and study hours (binned means with CI)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, col, bins, labs, ttl in [
        (axs[0], "attendance", [0, 60, 75, 90, 99.9, 100], ["<60", "60-75", "75-90", "90-99", "100"], "Class attendance %"),
        (axs[1], "study_hours", [-0.1, 1, 2, 3, 4, 6, 20], ["≤1", "1-2", "2-3", "3-4", "4-6", ">6"], "Study hours / day")]:
        t = d.dropna(subset=[col]).assign(b=lambda x: pd.cut(x[col], bins, labels=labs))
        a = t.groupby("b", observed=True).cgpa.agg(["mean", "sem", "count"])
        ax.errorbar(range(len(a)), a["mean"], yerr=1.96 * a["sem"], fmt="o-", color=P.CAT[0], lw=1.6, ms=6, mec=P.SURFACE, ecolor=P.INK2, capsize=2)
        ax.set_xticks(range(len(a))); ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(a.index, a["count"])], fontsize=8)
        ax.set_xlabel(ttl)
    axs[0].set_ylabel("Mean CGPA (4.0 scale), 95% CI"); axs[0].set_title("Attendance"); axs[1].set_title("Study time")
    P.save(fig, FIG / "A5_dose_response.png")
    a = d.dropna(subset=["attendance"]).groupby(pd.cut(d.attendance, [0, 60, 75, 90, 99.9, 100]), observed=True).cgpa.mean()
    res["attendance_bins"] = {str(k): round(float(v), 3) for k, v in a.items()}

    res["top_ols_m1"] = m1.params.drop("Intercept").abs().sort_values(ascending=False).head(6).round(3).to_dict()
    res["group_gaps"] = grp.set_index("factor")[["diff", "ci_lo", "ci_hi"]].round(3).to_dict("index")
    json.dump(res, open(TAB / "results_real.json", "w"), indent=2, default=float)
    return res
