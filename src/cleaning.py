"""Part A: load and clean the real CSE student survey (1,194 students, 4.0 CGPA scale)."""
import numpy as np
import pandas as pd
from .config import RAW, PROCESSED

RENAME = {
    "University Admission year": "admission_year",
    "Gender": "gender",
    "Age": "age",
    "Current Semester": "semester",
    "Do you have meritorious scholarship ?": "scholarship",
    "Do you use University transportation?": "uni_transport",
    "How many hour do you study daily?": "study_hours",
    "How many times do you seat for study in a day?": "study_sessions",
    "What is your preferable learning mode?": "learning_mode",
    "Do you use smart phone?": "smartphone",
    "Do you have personal Computer?": "own_pc",
    "How many hour do you spent daily in social media?": "social_hours",
    "Status of your English language proficiency": "english",
    "Average attendance on class": "attendance",
    "Did you ever fall in probation?": "probation",
    "Did you ever got suspension?": "suspension",
    "Do you attend in teacher consultancy for any kind of academical problems?": "consultancy",
    "How many hour do you spent daily on your skill development?": "skill_hours",
    "What is your relationship status?": "relationship",
    "Are you engaged with any co-curriculum activities?": "cocurricular",
    "With whom you are living with?": "living_with",
    "Do you have any health issues?": "health_issue",
    "What was your previous SGPA?": "prev_sgpa",
    "Do you have any physical disabilities?": "disability",
    "What is your current CGPA?": "cgpa",
    "How many Credit did you have completed?": "credits",
    "What is your monthly family income?": "family_income",
}


def load_and_clean():
    raw = pd.read_csv(RAW)
    raw.columns = [c.strip() for c in raw.columns]
    df = raw.rename(columns=RENAME)[list(RENAME.values())].copy()
    log = [("raw rows", len(df))]

    df["attendance"] = pd.to_numeric(df["attendance"], errors="coerce")
    n_bad = int(((df.attendance < 0) | (df.attendance > 100)).sum() + df.attendance.isna().sum())
    df.loc[(df.attendance < 0) | (df.attendance > 100), "attendance"] = np.nan
    log.append(("attendance missing/invalid -> NaN", n_bad))

    for col, cap in [("study_hours", 16), ("social_hours", 16), ("skill_hours", 16)]:
        bad = int((df[col] > cap).sum())
        df.loc[df[col] > cap, col] = np.nan
        log.append((f"{col} > {cap} h/day -> NaN", bad))
    bad = int((df.semester > 12).sum())
    df.loc[df.semester > 12, "semester"] = np.nan
    log.append(("semester > 12 -> NaN", bad))

    # CGPA of 0 = no grades earned yet (typically first-semester students): not a performance score
    n0 = int((df.cgpa <= 0).sum())
    df = df[df.cgpa > 0].copy()
    log.append(("CGPA <= 0 removed (no graded work yet)", n0))
    df["prev_sgpa"] = df["prev_sgpa"].where(df["prev_sgpa"] > 0)

    for col in ["health_issue"]:
        df[col] = df[col].astype(str).str.strip().str.lower().map(
            {"yes": "Yes", "no": "No", "n": "No"})
    df["relationship"] = np.where(df["relationship"].eq("Single"), "Single", "Partnered")
    df["log_income"] = np.log10(df["family_income"].clip(lower=1000))
    log.append(("clean rows", len(df)))
    df = df.reset_index(drop=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED / "cse_survey_clean.csv", index=False)
    return df, pd.DataFrame(log, columns=["step", "count"])
