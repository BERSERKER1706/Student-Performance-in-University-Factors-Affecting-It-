"""Run the whole project:  python run_all.py   (about a minute)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import FIG, TAB
from src.cleaning import load_and_clean
from src.analysis_real import run as run_real
from src.simulate import simulate
from src.analysis_india import run as run_india

if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    df, log = load_and_clean()
    log.to_csv(TAB / "A_cleaning_log.csv", index=False)
    print("Part A - real CSE survey\n", log.to_string(index=False))
    a = run_real(df)
    print(f"  n={a['n']}  OLS R2 without/with prior SGPA = {a['ols']['r2_without_prior']:.2f}/{a['ols']['r2_with_prior']:.2f}"
          f"  GBM CV R2 = {a['gbm']['r2_without_prior_mean']:.2f}/{a['gbm']['r2_with_prior_mean']:.2f}")
    sim = simulate()
    b = run_india(sim)
    print(f"Part B - simulated India cohort  n={b['n']}  planted effects recovered in 95% CI: {b['mixed']['coverage']:.0%}"
          f"  GBM CV R2 = {b['gbm_r2']['mean']:.2f}")
    print("Figures ->", FIG, "\nTables  ->", TAB)
