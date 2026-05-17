"""
Landmark variant of the dose-threshold sensitivity for Cohort B
(addresses the immortal-time bias concern raised about Sens-C).

HPV-vaccine standard schedule is 0–2–6 months for the three commercial
products (Gardasil 9, Gardasil 4-valent; Cervarix is 0–1–6). To complete
a k-dose schedule a patient must survive event-free until receipt of the
k-th dose. Conditioning on completion alone — as the original Sens-C
does — introduces immortal time between dose 1 and dose k.

The landmark variant addresses this by:
  1. Anchoring time at a landmark t* ≥ expected k-th-dose date + grace
  2. Restricting the analysis to patients alive AND event-free at t*
  3. For vaccinated cases, requiring ≥k doses BY t* (otherwise drop the
     entire matched set)
  4. Refitting Cox with time measured from t* (left-truncation)

Landmarks chosen:
  - ≥1 dose : t* = 30 d   (1-month grace from first dose)
  - ≥2 doses: t* = 90 d   (0–2 schedule + 1-month grace)
  - ≥3 doses: t* = 240 d  (0–2–6 schedule + 2-month grace)

Output: Data/Sensitivity_DoseThreshold_Landmark.csv
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data"

# --- Dose-date inventory per patient ---
print("Loading prescription file...")
rx = pd.read_csv(DATA / "한국 HPV 코호트 자료를 이용한 자_처방정보.csv",
                 encoding="cp949", low_memory=False)
mask = (rx["처방명"].astype(str).str.contains(
            "Gardasil|Cervarix|HPV vaccine", case=False, na=False) |
        rx["처방한글명"].astype(str).str.contains("가다실|서바릭스", na=False))
rx_vac = rx[mask].copy()
rx_vac["처방일자"] = pd.to_datetime(
    rx_vac["처방일자"].astype("Int64").astype(str),
    format="%Y%m%d", errors="coerce")

# For each patient: sorted dose dates
doses = (rx_vac.sort_values(["연구번호", "처방일자"])
              .groupby("연구번호")["처방일자"]
              .apply(list)
              .to_dict())


def kth_dose_date(pid: str, k: int):
    """Return date of the k-th dose for `pid` or NaT if fewer than k doses."""
    dates = doses.get(pid, [])
    if len(dates) < k:
        return pd.NaT
    return dates[k - 1]


# --- Cohort B (lesion recurrence) and clearance subset ---
B = pd.read_csv(DATA / "final_matched_outcomes.csv", encoding="utf-8-sig")
B["vac"] = B["접종여부"].astype(bool).astype(int)
B["index_date"] = pd.to_datetime(B["index_date"])
B["recurrence_date"] = pd.to_datetime(B["recurrence_date"], errors="coerce")
B["follow_up_days"] = pd.to_numeric(B["follow_up_days"], errors="coerce")
B["index_age"] = pd.to_numeric(B["index_age"], errors="coerce")
B["has_recurrence"] = B["has_recurrence"].astype(int)
B["days_to_recurrence"] = pd.to_numeric(B["days_to_recurrence"], errors="coerce")

BC = pd.read_csv(DATA / "CohortB_Clearance_Analytic.csv", encoding="utf-8-sig")
BC["vac"] = BC["vac"].astype(int)
BC["index_date"] = pd.to_datetime(BC["index_date"])
BC["first_neg_date"] = pd.to_datetime(BC["first_neg_date"], errors="coerce")
BC["follow_up_days"] = pd.to_numeric(BC["follow_up_days"], errors="coerce")
BC["index_age"] = pd.to_numeric(BC["index_age"], errors="coerce")
BC["has_clearance"] = BC["first_neg_date"].notna().astype(int)
BC["days_to_clear"] = (BC["first_neg_date"] - BC["index_date"]).dt.days


def fit_landmark(df, ev_col, time_col, k: int, landmark_days: int) -> dict:
    """Apply landmark restriction at `landmark_days` and dose threshold k,
    then fit age-adjusted Cox with cluster-robust SE on fine_match_id.

    Time is left-truncated at the landmark: the at-risk clock starts at
    `landmark_days` and only events occurring after that point are counted.
    """
    d = df.copy()
    d["event_time"] = np.where(d[ev_col].astype(bool), d[time_col],
                               d["follow_up_days"])

    # --- Step 1: vaccinated case must have ≥k doses BY the landmark ---
    if k > 1:
        kth = d["연구번호"].map(lambda p: kth_dose_date(p, k))
        days_to_kth = (kth - d["index_date"]).dt.days
        # Vaccinated cases failing the criterion → drop their matched sets
        bad = set(d.loc[(d["vac"] == 1) &
                        (days_to_kth.isna() | (days_to_kth > landmark_days)),
                        "fine_match_id"])
        d = d[~d["fine_match_id"].isin(bad)].copy()

    # --- Step 2: everyone must be alive and event-free at the landmark ---
    # i.e., event_time > landmark_days (events before landmark are dropped)
    d = d[d["event_time"] > landmark_days].copy()

    # --- Step 3: left-truncated time = event_time - landmark_days ---
    d["time"] = d["event_time"] - landmark_days

    df_fit = d[["time", ev_col, "vac", "index_age", "fine_match_id"]].rename(
        columns={ev_col: "event"}).dropna()
    df_fit["event"] = df_fit["event"].astype(int)
    df_fit = df_fit[df_fit["time"] > 0]

    n_v = int((df_fit["vac"] == 1).sum())
    n_c = int((df_fit["vac"] == 0).sum())
    e_v = int(((df_fit["vac"] == 1) & (df_fit["event"] == 1)).sum())
    e_c = int(((df_fit["vac"] == 0) & (df_fit["event"] == 1)).sum())
    res = dict(n_v=n_v, n_c=n_c, ev_v=e_v, ev_c=e_c,
               HR=np.nan, CIlo=np.nan, CIhi=np.nan, p=np.nan)
    if e_v + e_c < 3 or n_v < 2 or n_c < 2:
        return res
    try:
        cph = CoxPHFitter().fit(df_fit, duration_col="time", event_col="event",
                                cluster_col="fine_match_id", robust=True)
        r = cph.summary.loc["vac"]
        res.update(HR=float(r["exp(coef)"]),
                   CIlo=float(r["exp(coef) lower 95%"]),
                   CIhi=float(r["exp(coef) upper 95%"]),
                   p=float(r["p"]))
    except Exception as e:
        print(f"    Cox fit failed for k={k}, landmark={landmark_days}: {e}")
    return res


LANDMARKS = [(1, 30,  "≥1 dose, 30-d landmark"),
             (2, 90,  "≥2 doses, 90-d landmark"),
             (3, 240, "≥3 doses, 240-d landmark")]

rows = []
for ev_label, df, ev_col, time_col in [
    ("Lesion recurrence", B,  "has_recurrence", "days_to_recurrence"),
    ("hr-HPV clearance",  BC, "has_clearance",  "days_to_clear"),
]:
    for k, landmark, defn in LANDMARKS:
        r = fit_landmark(df, ev_col, time_col, k, landmark)
        r.update(outcome=ev_label, definition=defn,
                 threshold=k, landmark_days=landmark)
        rows.append(r)
        if not np.isnan(r["HR"]):
            print(f"  {ev_label:18s} {defn:30s} "
                  f"n_v={r['n_v']:>3}/n_c={r['n_c']:>3}  "
                  f"events {r['ev_v']}/{r['ev_c']}  "
                  f"HR={r['HR']:.2f} ({r['CIlo']:.2f}–{r['CIhi']:.2f})  p={r['p']:.3f}")
        else:
            print(f"  {ev_label:18s} {defn:30s} insufficient events")

out = pd.DataFrame(rows)[
    ["outcome", "definition", "threshold", "landmark_days",
     "n_v", "n_c", "ev_v", "ev_c", "HR", "CIlo", "CIhi", "p"]]
out_path = DATA / "Sensitivity_DoseThreshold_Landmark.csv"
out.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"\nSaved: {out_path.relative_to(ROOT)}")
