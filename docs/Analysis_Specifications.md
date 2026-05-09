# Analysis specifications — Index dates, outcomes, cohorts

A single-source reference for every analysis reported in the HPV-vaccine
manuscript. Each entry specifies the eligibility filter, the index date
convention, the outcome definition, the matched-set integrity rule, and
the source script and result file.

---

## 1. Source data

| Asset | File | Encoding | Notes |
|---|---|---|---|
| Source population | `Data/한국 HPV 코호트 자료를 이용한 자_코호트.csv` | cp949 | N = 32,969; one row per study ID. PHI; gitignored. |
| Prescription | `Data/한국 HPV 코호트 자료를 이용한 자_처방정보.csv` | cp949 | All prescription orders. HPV vaccine identified by drug-name keywords (`Gardasil`/`Cervarix`/`HPV vaccine`/`가다실`/`서바릭스`) AND drug-code prefixes (`DV-9HPF`/`DV-HPF`/`DV-JHP`); both definitions concord at 2,156 patients. PHI; gitignored. |
| Diagnosis (with comorbidity tag) | `Data/한국 HPV 코호트 자료를 이용한 자_진단정보_기저질환추가_unlocked.xlsx` | – | Pre-classified into 5 chronic-condition groups (Angina/MI, HTN, DM, Stroke, PE). PHI; gitignored. |
| Surgery | `Data/한국 HPV 코호트 자료를 이용한 자_수술처방_수술종류구분완료.csv` | cp949 | Manually labelled `수술 종류` field: `1` = conization, `3` = hysterectomy, `제외` = excluded. PHI; gitignored. |
| Pathology | `Data/한국 HPV 코호트 자료를 이용한 자_병리검사 (복구됨).CSV` | cp949 | Free-text 판독결과 column. Sub-stratified by `병리검사구분`: `조직병리` (tissue) for lesion recurrence, `분자병리`/`HPV` (molecular) for HPV outcomes. PHI; gitignored. |
| Clinical info | `Data/한국 HPV 코호트 자료를 이용한 자_기초임상정보.csv` | cp949 | Height/weight/BP/smoking, with `기록일자`. PHI; gitignored. |

All scripts use `random seed = 42`.

---

## 2. Exposure definition (HPV vaccination)

- **Source**: prescription file.
- **Identification**: a record qualifies as an HPV-vaccine dose if either
  - `처방명` matches `/Gardasil|Cervarix|HPV vaccine/i`, **or**
  - `처방한글명` matches `/가다실|서바릭스/`, **or**
  - `처방코드` starts with `DV-9HPF` (Gardasil 9), `DV-HPF` (Gardasil 4-valent), or `DV-JHP` (Cervarix), including the `-FR` (free-of-charge) and `-FJ` (employee/family) variants and the legacy `DV-HPJ` code.
- **First-vaccine date** per patient = earliest qualifying `처방일자`.
- **Vaccinated**: any record satisfies the above (n = 2,156).
- **Vaccine-type attribution**: when a patient received doses of more than one product (33 of 2,156, 1.5%), the type of the **first dose** is used.
- **Dose count**: number of qualifying records per patient. Median 3 (IQR 2–3, range 1–6); 80% completed ≥3 doses.

Validation: `scripts/sensitivity_exposure_definition.py` (S2 — prescription-code cross-check; S3 — mixed-vaccine pattern; outputs `Data/Sensitivity_PrescriptionCode.csv` and `Data/Sensitivity_MixedVaccineType.csv`).

---

## 3. Cohort A — Long-term safety analysis

### 3.1 Eligibility

- All women in the source population (N = 32,969) with
  - non-missing `생년월` (birth date),
  - alive at the index date,
  - ≥ 1 day of follow-up after the index date.

### 3.2 Index date

| Group | Index date |
|---|---|
| Vaccinated | First HPV-vaccine prescription date (`first_vaccine_date`). |
| Unvaccinated | Pseudo index date assigned by random sampling (seed 42) from the empirical distribution of first-vaccine dates. |

### 3.3 Matching

- **Algorithm**: 1:1 nearest-neighbour matching on logit(propensity score).
- **Caliper**: 0.2 × SD of logit(PS), without replacement (Austin 2011).
- **Propensity-score model** (logistic regression):
  - age at index, BMI, SBP, DBP (mean imputation + missing indicators each), smoking (Never / Former / Current / Unknown reference), residence in Seoul.
- **Final n** = 4,102 (2,051 vaccinated / 2,051 unvaccinated).

### 3.4 Outcomes

Five pre-classified chronic conditions plus two pre-specified composites:

| Outcome | Operational definition |
|---|---|
| Angina/MI (`I20–I25`) | First post-index diagnosis date with classification tag `1`. |
| Hypertension (`I10–I15`) | First post-index date with tag `2`. |
| Diabetes (`E11–E14`) | First post-index date with tag `3`. |
| Stroke (`I60–I69`) | First post-index date with tag `4`. |
| Pulmonary embolism (`I26`) | First post-index date with tag `5`. |
| **Any-of-5 composite** | First post-index occurrence of any of the five. |
| **MCE (Major Cardiovascular Events)** | First post-index occurrence of MI (1) or Stroke (4) or PE (5). |

Prevalent disease (any of the five tagged with `diag_date ≤ index_date`) excludes the patient from that specific outcome's analysis. Death is treated as a competing event.

### 3.5 Pre-specified sensitivity analyses (Cohort A)

| ID | Description | Output |
|---|---|---|
| – | Aalen–Johansen CIF for each outcome (cause-specific). | `Data/Figure2_CohortA_CIF_HR.png` |
| – | Fine–Gray subdistribution model with Geskus IPCW reweighting. | (in same figure / `Data/Table2_CohortA_HazardRatios.csv`) |
| **S8** | Pseudo-index assignment sensitivity (3 strategies: random sampling, calendar-year-matched, risk-set sampling). | `Data/CohortA_pseudoindex_sensitivity.csv` |
| **S9** | Dose-threshold (≥1 / ≥2 / ≥3 doses), matched-pair preserved. | `Data/Sensitivity_DoseThreshold_HR.csv` |
| **S2** | Prescription-code validation. | `Data/Sensitivity_PrescriptionCode.csv` |

---

## 4. Cohort B — Post-surgical efficacy analysis

### 4.1 Eligibility

- Women in the source population who underwent cervical conization or hysterectomy for histologically confirmed HSIL/CIN3+ disease (N_pool = 6,890).
- After matching: index date ≤ 31 December 2020, ≥ 2 follow-up records.

### 4.2 Index date

| Group | Index date |
|---|---|
| Vaccinated | First HPV-vaccine prescription date AFTER the qualifying surgery (`identify_vaccinated_group` requires `접종일자 > 첫수술일자`). |
| Unvaccinated | Pseudo index date = `첫수술일자 + 매칭된 접종군의 수술-접종 간격(T)`. This preserves the immortal-time correspondence between groups. |

### 4.3 Matching

Two-step variable-ratio greedy matching, **without replacement**. Requested ratios are *upper bounds*; if fewer comparable controls exist, fewer are used.

| Step | Variables (tolerance) | Ratio | Realised mean | Output |
|---|---|---|---|---|
| Step 1 — initial | surgery method (exact), surgery year (±1 y), age at surgery (±5 y) | 1:up-to-5 | 4.42 (256 of 411 cases hit the cap) | 411 vac / 1,815 ctl |
| Step 2 — index-date filter | index date ≤ 2020-12-31, ≥ 2 follow-up records | – | – | 411 / 1,797 (18 ctl excluded) |
| Step 3 — fine matching | index age (±5 y), BMI nearest index (±3 kg/m²), surgery year (±1 y); BMI condition relaxed if no candidate has BMI | 1:up-to-4 | 3.60 (193 of 241 cases hit the cap) | 241 / 867 (n = 1,108) |

### 4.4 Co-primary outcomes

#### P1. Lesion recurrence (≥CIN2 / HSIL+ or invasive cervical cancer)

- **Source**: `병리검사구분 == '조직병리'` records with `실시일자 > index_date`.
- **Detection** (`detect_hsil_cin3_recurrence`, regex on 판독결과; in priority order):
  1. Carcinoma — squamous cell carcinoma, adenocarcinoma, cervical cancer, invasive carcinoma, carcinoma in situ (`CIS`).
  2. CIN3 / CIN III / CINIII / CIN-3.
  3. HSIL / H-SIL / high-grade squamous intraepithelial lesion / high-grade SIL.
  4. CIN2 / CIN II / CINII / CIN-2 / CIN 2/3 / CIN II/III.
- **Event date**: first qualifying record per patient (`recurrence_date`).
- **Direction**: HR < 1 favours vaccination.
- **Cohort**: full Cohort B (n = 1,108). Events 13 vac / 57 non-vac.

#### P2. hr-HPV clearance / regression

- **Cohort**: subset of Cohort B with documented pre-vaccine hr-HPV positivity (n = 292: 110 vac / 182 non-vac after matched-set integrity filter).
  - **Pre-vaccine baseline** = union of hr-HPV types detected on any molecular pathology record with `실시일자 < index_date`. The pre-vaccine baseline (rather than pre-surgery) anchors the temporal frame to the start of exposure for both arms (the pseudo-index for unvaccinated controls).
  - **Matched-set integrity**: drop fine_match_ids whose vaccinated case lacks the qualifying baseline; drop non-vaccinated members who themselves lack a pre-vaccine HPV+ test.
- **Source**: `병리검사구분 ∈ {'분자병리','HPV'}` records with `실시일자 > index_date`.
- **Event**: date of the FIRST of two **consecutive** post-index molecular pathology records explicitly negative for hr-HPV (`detect_high_risk_hpv` returns `is_high_risk_hpv_positive = False` for two adjacent records by date). The two-consecutive-negative requirement follows Bouvard (2009) and Insinga (2010), guarding against single-negative misclassification from imperfect assay sensitivity (~5–10% false-negative rate) or transient viral-load fluctuation.
- **Event date**: `first_neg_date` (= first of the two consecutive negatives).
- **Direction**: **HR > 1** favours vaccination (faster clearance).
- Events 40 vac / 48 non-vac.
- A single-negative-test definition is reported as a sensitivity (Sens-M, see §4.5).

### 4.5 Sensitivity outcomes (Cohort B)

| ID | Outcome | Cohort | Event | Time anchor | Direction | Output |
|---|---|---|---|---|---|---|
| Sens-A | Post-index hr-HPV detection (any) | Full Cohort B (n = 1,108) | First post-index molecular record with `is_high_risk_hpv_positive = True` | Index | HR < 1 | `Data/CohortB_HR_revised.csv` row 3 |
| Sens-B | Post-index detection landmark (6/12/24 mo) | Patients still at risk at landmark | First HPV+ after landmark | Landmark date | HR < 1 | `Data/Sensitivity_HPV_Landmark.csv` |
| Sens-C | Novel-type acquisition | Pre-vaccine HPV-test available; matched-set integrity | First post-index record with type **not** in pre-vaccine type set | Index | HR < 1 | `Data/Sensitivity_HPV_NovelType.csv` |
| Sens-D | HPV-16 clearance | Pre-vaccine HPV-16+ baseline | First post-index molecular record without HPV 16 | Index | HR > 1 | `Data/Sensitivity_HPV_Clearance.csv` C2 |
| Sens-E | HPV-18 clearance | Pre-vaccine HPV-18+ baseline | First post-index molecular record without HPV 18 | Index | HR > 1 | `Data/Sensitivity_HPV_Clearance.csv` C3 |
| Sens-F | Dose threshold (≥2 / ≥3) | Drop matched sets where vac case fails threshold | Same as primary | Index | HR < 1 | `Data/Sensitivity_DoseThreshold_HR.csv` |
| Sens-G | Strict 1:4 fine matching | Drop matched sets where vac case has < 4 controls | Same as primary | Index | HR < 1 (recurrence) | `Data/Sensitivity_StrictMatching.csv` |
| Sens-H | Vaccine-type interaction (LRT) on each outcome | Full or clearance subset depending on outcome | Same as primary | Index | – | `Data/CohortB_vaccine_interaction.csv` |
| Sens-I | Vaccine-type × calendar period | Strata of clearance subset by index year | Clearance | Index | HR > 1 | `Data/Sensitivity_VaccineType_ByCalendar.csv` |
| Sens-J | Restricted follow-up (3-y / 5-y) and unadjusted | Full cohort or clearance subset | Same as primary, censored | Index | – | `Data/sensitivity_analysis_both_outcomes.csv` |
| Sens-K | Age-stratified lesion recurrence × FU window | Full Cohort B by age stratum | Lesion recurrence censored at window | Index | HR < 1 | `Data/CohortB_age_fu_forest.csv` |
| Sens-L | Time-stratified clearance HR (0–6, 6–12, 12–24, ≥24 mo) | Clearance subset | Same as P2; restricted to events within each window | Index + window lower bound | HR > 1 | `Data/Sensitivity_HPV_Clearance_TimeStratified.csv` |
| Sens-M | Single-negative clearance (alternative to primary two-negative) | Clearance subset | First single post-index hr-HPV-negative record | Index | HR > 1 | `Data/Sensitivity_HPV_Clearance_SingleNegative.csv` |
| Sens-N | Lesion recurrence with minimum disease-free interval (3 / 6 / 12 mo) | Full Cohort B, restricted to those without recurrence before the minimum interval | Lesion recurrence after the minimum interval | Index + minimum interval | HR < 1 | `Data/Sensitivity_Recurrence_DFInterval.csv` |

### 4.6 Statistical model (both co-primary outcomes)

- **Cox proportional-hazards** with **age at index** as the only adjustment covariate.
- **Cluster-robust standard errors** clustered on `fine_match_id`.
- **Time** = `days_to_event` if event observed (i.e. `days_to_recurrence` for P1 and `(first_neg_date − index_date).days` for P2), else `follow_up_days = 최종추적일자 − index_date`.
- **Censoring** at last follow-up, death, loss to follow-up, or 31 December 2025, whichever first.
- **Schoenfeld residual rank test** reported on the vaccinated covariate; lesion recurrence p = 0.82 (no PH violation), hr-HPV clearance p = 0.007 (interpreted as biologically expected non-constancy of any vaccine effect on clearance over follow-up).

---

## 5. Quick-reference summary table

| | Cohort A — Safety | Cohort B (recurrence co-primary) | Cohort B (clearance co-primary) |
|---|---|---|---|
| Final n | 4,102 | 1,108 | 292 |
| Vac / Non-vac | 2,051 / 2,051 | 241 / 867 | 110 / 182 |
| Match | PSM 1:1, caliper 0.2 SD logit(PS) | Variable-ratio 1:up-to-5 → 1:up-to-4 | Same as recurrence + pre-vaccine HPV+ filter |
| Index (vac) | First vaccine date | First vaccine date after surgery | First vaccine date after surgery |
| Index (non-vac) | Random pseudo-date | Surgery + matched interval (T) | Surgery + matched interval (T) |
| Event source | Diagnosis records (5 chronic) | Tissue pathology (조직병리) | Molecular pathology (분자병리/HPV) |
| Event definition | First post-index ICD-10 group hit | First post-index ≥CIN2 (HSIL+) | First post-index hr-HPV-NEG record |
| Favourable HR direction | HR < 1 | HR < 1 | **HR > 1** |
| Primary HR (95% CI) | 1.26 (0.75–2.12) Any-of-5 | 0.80 (0.44–1.43) | 1.40 (0.92–2.11) |
| p value | 0.37 | 0.45 | 0.11 |

All point estimates non-significant at the conventional α = 0.05 threshold.

---

## 6. Reproducibility map

| Asset | Generator |
|---|---|
| Cohort A matched dataset (in-memory) | `scripts/make_main_figures.py::build_cohort_a_matched` |
| Cohort B matched dataset | `scripts/build_matched_cohort.py` → `scripts/build_final_cohort.py` |
| Outcome ascertainment | `scripts/extract_pathology_outcomes.py`, `scripts/extract_outcomes_after_index.py` |
| Cohort A primary results | `scripts/cohort_a_psm_hr_cif.py` |
| Cohort B co-primary results | `scripts/analyze_cohortB_clearance_primary.py`, `scripts/rebuild_table3.py` |
| Figure 1 (cohort flow) | `scripts/make_main_figures.py::figure1`, PPT via `scripts/make_figure1_pptx.py` |
| Figure 2 (Cohort A CIF + forest) | `scripts/make_main_figures.py::figure2` |
| Figure 3 (Cohort B co-primary CIF) | `scripts/make_main_figures.py::figure3` |
| Figure 4 (Cohort B subgroup forest) | `scripts/make_main_figures.py::figure4_subgroup` |
| Sup S1 / S2 (love plots, PS density) | `scripts/regenerate_love_plots.py` |
| Sup S6 / S7 / S14 (vaccine-type interaction etc.) | `scripts/rebuild_supplementary_clearance.py` |
| Sup S3 / S4 / S15 / S16 | `scripts/rebuild_supplementary_misc.py` |
| Sup S9 (dose threshold), S2 (Rx-code), S12 (mixed) | `scripts/sensitivity_exposure_definition.py` |
| Sup S10 (strict matching) | `scripts/sensitivity_strict_matching.py` |
| Sup S13 (HPV landmark) | `scripts/sensitivity_hpv_landmark.py` |
| Sup S15 (novel-type stand-alone) | `scripts/sensitivity_hpv_novel_type.py` |
| Sup S16 (clearance stand-alone) | `scripts/sensitivity_hpv_clearance.py` |
| Sup S14 (vacc × calendar) stand-alone | `scripts/sensitivity_vaccine_type_calendar.py` |
| Table 1 (baseline characteristics) | `scripts/baseline_table1_unified.py`, `scripts/append_table1_clearance_subset.py` |
| Table 3 (Cohort B HR) | `scripts/rebuild_table3.py` |

---

## 7. Convention checklist (for manuscript reviewers)

- [x] All matching is **without replacement** with **random seed = 42**.
- [x] Variable-ratio matching is consistently described as `1:up-to-N` with mean realised ratios.
- [x] Cohort B baseline for HPV outcomes is the **pre-vaccine** baseline (records with `실시일자 < index_date`), not the pre-surgery baseline.
- [x] Cox time = `days_to_event` for events, `follow_up_days` otherwise.
- [x] Cluster-robust SE uses `pair_id` (Cohort A) or `fine_match_id` (Cohort B).
- [x] Direction of effect is explicitly stated for every reported HR (favourable HR < 1 vs > 1).
- [x] Lesion recurrence outcome is CIN2+ (≥CIN2 / HSIL+ or invasive carcinoma); surgical eligibility was HSIL/CIN3+ (the standard surgical indication, distinct from the post-index outcome threshold).
- [x] All sensitivity tables (S1–S16) are consistent with the manuscript narrative as of git commit `2c38329`.
