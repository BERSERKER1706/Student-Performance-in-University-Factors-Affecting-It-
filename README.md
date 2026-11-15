# Student Performance in University: Factors Affecting It

A Python project that asks what actually explains a university student's CGPA, and how much of it can be predicted. It has two parts:

- **Part A: real survey data.** A cleaned survey of 1,155 students, analysed with correlations, group comparisons, regression and machine-learning models.
- **Part B: simulated Indian multi-branch cohort.** A clearly labelled synthetic dataset of 3,500 students with *known, planted* effects. It checks whether the statistical methods can recover the truth, and lets us compare branches and institution tiers that the real survey does not cover.

This is a follow-up to the earlier BKBIET data-visualization report, which used purely simulated data. Here the real-data findings stand on their own, and the simulated data is kept separate and labelled.

---

## Key results

### Part A: real data (n = 1,155, mean CGPA 3.27)

| Model | R² without previous SGPA | R² with previous SGPA |
|---|---|---|
| Linear regression (OLS) | 0.09 | 0.58 |
| Gradient boosting (cross-validated) | 0.05 | 0.69 |

- **Past performance dominates.** Adding previous-semester SGPA lifts explained variance from under 10% to 58–69%. Without it, the surveyed habits and background factors explain very little.
- **Without prior SGPA**, the significant predictors are attendance (+), English score (+), social hours (−) and living arrangement. Several other factors were not distinguishable from zero.
- **With prior SGPA** controlled, prior SGPA is the largest effect (coef ≈ 0.40). English score, family income (log), semester, relationship status, age (−) and consultancy (−) also come out significant.
- **Self-reported daily study hours showed no significant relationship** with CGPA in either model (p ≈ 0.6 and 0.7).

### Part B: simulated cohort (n = 3,500, 10 institutions)

| Check | Result |
|---|---|
| Planted effects recovered inside the 95% interval | **94%** (17 of 18) |
| Prediction R² (gradient boosting) | 0.60 |
| Share of CGPA variance between institutions (ICC) | 0.05 |

The one effect outside its interval was study hours per day (true 0.20, estimated 0.246). Class 12 %, attendance, stress, screen time and sleep were recovered very closely. Branch effects were estimated relative to EE (for example, DS and CSE above EE, ME below).

> **Caution:** Part B conclusions describe what the *method* can detect in synthetic data. They are not evidence about real students.

---

## Data cleaning (Part A)

| Step | Rows affected |
|---|---|
| Raw rows | 1,194 |
| Attendance missing or invalid, set to NaN | 1 |
| Social hours above 16 h/day, set to NaN | 1 |
| Semester above 12, set to NaN | 36 |
| CGPA ≤ 0 removed (no graded work yet) | 39 |
| **Clean rows** | **1,155** |

The full log is saved to `A_cleaning_log.csv`.

---

## Project structure

```
student_performance_project/
├── src/
│   ├── config.py          # paths (FIG, TAB output folders)
│   ├── cleaning.py        # load_and_clean(): cleaning pipeline + log
│   ├── analysis_real.py   # Part A: correlations, group gaps, OLS, GBM, dose-response
│   ├── simulate.py        # Part B: synthetic cohort generator
│   └── analysis_india.py  # Part B: mixed model, recovery check, GBM
├── Students_perfromnace.ipynb   # runs everything end to end
├── figures/               # A1–A5, B1–B4 charts (PNG)
└── tables/                # CSV outputs (cleaning log, OLS coefficients, recovered vs true)
```

> Folder names for `figures/` and `tables/` follow `src/config.py` (`FIG`, `TAB`). Adjust this tree if your layout differs.

## Outputs

**Figures:** `A1_correlations`, `A2_group_gaps`, `A3_regression`, `A4_importance`, `A5_dose_response` (real data); `B1_branch_cgpa`, `B2_recovered_vs_true`, `B3_study_by_branch`, `B4_importance` (simulated).

**Tables:** `A_cleaning_log.csv`, `A_ols_coefficients.csv` (coefficients, 95% CIs and p-values with and without prior SGPA), `B_recovered_vs_true.csv`.

## Methods

- **Part A:** correlation analysis, group gap comparisons, OLS regression (two specifications: with and without previous SGPA), gradient-boosting prediction with cross-validation, feature importance and dose-response curves.
- **Part B:** a mixed-effects model with institution as a grouping factor (reports ICC and checks 95% interval coverage against the planted effects), gradient-boosting prediction and feature importance.

## How to run

1. Install requirements (Python 3.10+):
   ```bash
   pip install pandas numpy scipy statsmodels scikit-learn matplotlib
   ```
2. Open `Students_perfromnace.ipynb` in Jupyter. The first cells locate the project root (the folder containing `src/`), so it works from any launch directory.
3. Run all cells in order. Figures and tables are written to the output folders.

## Limitations

- Survey data is self-reported and cross-sectional, so results show association, not causation.
- Study hours are self-reported and may be noisy, which can weaken their apparent effect.
- Part A explains little without prior SGPA, so important drivers (prior ability, teaching quality, motivation and so on) are not captured by the survey.
- Part B is synthetic by construction; its effects are known because they were planted.

## Tech stack

Python, pandas, NumPy, SciPy, statsmodels, scikit-learn, Matplotlib, Jupyter.

## Author

**Sir Skull** — B.Tech Computer Science, B.K. Birla Institute of Engineering and Technology, Pilani
GitHub: [BERSERKER1706](https://github.com/BERSERKER1706)
