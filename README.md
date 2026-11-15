# Performance of University Students and the Factors Affecting It

A two-part Python project on the question *"how do students perform at university, and what affects it?"*

| | Part A: real data | Part B: simulated data |
|---|---|---|
| **Data** | Real survey of 1,194 B.Sc. CSE students (1,155 after cleaning). 4.0 CGPA scale. Public Kaggle dataset, CC BY-NC-SA 4.0, mirrored in the `edupulse-datasets` GitHub repo. | 3,500 **simulated** students, 12 institutions in 3 tiers, 7 branches (CSE, DS, AI, EE, ECE, ME, CE), 10-point CGPA. |
| **Question** | What is actually associated with CGPA, and how much can be predicted? | Do standard methods recover known effects, and does the payoff of study time differ by branch? |
| **Status** | Real students, but self-reported, one university, **not Indian** (the 4.0 scale and "H.S.C." wording point to a South Asian university outside India). | Effects are **assumptions** declared in `src/config.py`. Nothing in Part B is a finding about real students. |

## Why the hybrid
I could not find a *verified, real* Indian multi-branch dataset reachable from the build environment. The candidates I did find (`engineering_student_journey`, `student_performance_tamilnadu`) behave like random data: GPA correlates about 0.00 with everything. The repo's own README says provenance is unverified. So the project uses the one real dataset with genuine signal for the real analysis, and a clearly labelled simulation for the Indian multi-branch design. **If you obtain a real Indian dataset, `src/analysis_india.py` runs on any CSV with the same column names.**

## Run it
```bash
pip install -r requirements.txt
python run_all.py        # about 25 seconds; writes outputs/figures and outputs/tables
```

## Part A findings (real CSE survey, n = 1,155)
- **Prior results dominate.** Previous-semester SGPA: Spearman ρ = 0.62 with CGPA.
- **Attendance is the one habit that stands out.** Mean CGPA is 3.00 for <60% attendance and 3.39 for 100% (ρ = 0.25). After controlling for prior SGPA the adjusted effect shrinks from 0.089 to 0.023 CGPA per SD (95% CI 0.001 to 0.044), so most of the attendance link is shared with earlier performance.
- **Study hours barely register.** ρ = 0.04, adjusted effect 0.008 per SD (95% CI −0.024 to 0.040). This is self-reported time, not quality of study.
- **Habits alone predict little.** Cross-validated R² without prior SGPA: gradient boosting 0.05 (±0.07), OLS 0.09 in-sample. With prior SGPA: 0.69 and 0.58.
- **Probation (−0.36), suspension (−0.33) and merit scholarship (+0.22) are results of performance, not causes**, so they are shown separately and kept out of the factor models.
- **Social media's sign flips.** Unadjusted −0.040 per SD, but +0.028 after controlling for prior SGPA. Treat it as unresolved, not as a finding.
- Preferring online learning (−0.10) and living away from family (−0.08) show gaps, and both weaken after adjusting for prior SGPA.

## Part B findings (simulated, n = 3,500)
- **Recovery check:** a mixed model (random intercept per institution) puts **17 of 18** planted effects inside its 95% CI. The one miss is study hours (estimate 0.25 vs planted 0.20). That is partly a comparison artifact: the true average slope across branches is about 0.27 because the planted branch interaction raises it, and the main-effect model does not include it.
- **Study time pays off differently by branch** (interaction test p = 0.003). The estimates run from about 0.19 (ME) to 0.32 (ECE) CGPA per SD, matching the planted pattern. This was built in, so it shows the method works, not that engineering branches behave this way.
- Institution explains about 5% of CGPA variance (ICC 0.049).
- Gradient boosting reaches CV R² ≈ 0.60. Class 12 %, attendance and study hours lead the permutation importances.

## Limitations
- Part A is observational and self-reported. Associations are not causes.
- Part A is one program at one university, so it cannot support claims about Indian universities or other branches.
- Part B's numbers come from my chosen assumptions. Branch offsets in particular are not evidence about real branch differences.
- Missing data is handled by listwise deletion in the models and median imputation in the machine-learning step.

## Layout
```
run_all.py            one-command pipeline
src/config.py         paths, seed, planted effects for the simulation
src/cleaning.py       Part A cleaning, with a cleaning log
src/analysis_real.py  Part A: correlations, group gaps, OLS (robust SE), gradient boosting, permutation importance
src/simulate.py       Part B generator
src/analysis_india.py Part B: mixed model, recovery check, branch heterogeneity, importance
data/raw              the real survey CSV
data/processed        cleaned survey, simulated cohort
outputs/figures       9 PNG charts (A1-A5, B1-B4)
outputs/tables        CSV and JSON results
```

## Data sources
- Student Academic Performance Dataset for ML (Kaggle, dhruvb2028), via the [edupulse-datasets](https://github.com/thinkter/edupulse-datasets) repository. License CC BY-NC-SA 4.0: **non-commercial use with attribution**.
