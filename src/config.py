"""Paths, seeds and the simulation's ground-truth parameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "student_academic_performance_cse.csv"
PROCESSED = ROOT / "data" / "processed"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
SEED = 2026
N_BOOT = 2000

# ---- Part B: ground truth planted in the simulated Indian multi-branch cohort ----
# Effects are on the 10-point CGPA scale per +1 SD of the (standardised) driver.
# They are ASSUMPTIONS chosen to be plausible in direction/size, not findings.
BRANCHES = ["CSE", "DS", "AI", "EE", "ECE", "ME", "CE"]
BRANCH_SHARE = [0.22, 0.08, 0.08, 0.14, 0.18, 0.16, 0.14]
BRANCH_OFFSET = {"CSE": 0.10, "DS": 0.15, "AI": 0.12, "EE": 0.0, "ECE": 0.02, "ME": -0.08, "CE": -0.05}
N_INSTITUTIONS = 12
N_STUDENTS = 3500
TRUE = {
    "class12_pct": 0.45,      # prior attainment
    "attendance": 0.30,
    "study_hours": 0.20,
    "sleep_hours": 0.08,
    "screen_hours": -0.12,    # recreational screen / social media
    "stress": -0.15,
    "paid_work": -0.10,       # part-time work
    "mentor_access": 0.10,
    "first_gen": -0.06,
    "hostel": -0.03,
    "tier_2": -0.15,          # relative to tier 1
    "tier_3": -0.30,
    "noise_sd": 0.55,
}
# study hours matter more in the heavy-maths branches (interaction, per +1 SD)
STUDY_BOOST = {"CSE": 0.0, "DS": 0.08, "AI": 0.08, "EE": 0.10, "ECE": 0.10, "ME": 0.04, "CE": 0.0}
