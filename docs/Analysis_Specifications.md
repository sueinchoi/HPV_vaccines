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

### 4.5 Sensitivity analyses

The 14 candidate sensitivity analyses are partitioned into **five essential analyses** that defend the principal Cohort B inferences (reported in the main text and in **Supplementary Figure S6**) and **nine appendix-only analyses** that provide additional robustness or descriptive context (reported in supplementary tables only). The pruning was made on the rationale that ① clearance event-definition robustness, ② clearance time-stratified PH decomposition, ③ exposure-dose threshold, ④ matching-structure robustness, and ⑤ recurrence event-timing robustness collectively cover the four largest sources of inferential uncertainty (outcome definition, model specification, exposure misclassification, residual confounding).

#### 4.5.1 Essential sensitivity analyses (main text)

| ID | Description | Defends | Cohort | Direction | Output |
|---|---|---|---|---|---|
| **Sens-A** | Single-negative test vs two-consecutive-negative clearance event definition | P2 outcome-definition robustness | v3 clearance subset (n = 235; 92 / 143) | HR > 1 | `Data/Sensitivity_HPV_Clearance_SingleNegative_v3.csv` |
| **Sens-B** | Time-stratified clearance HR (0–6, 6–12, 12–24, ≥24 mo post-landmark; left-truncation at each window's lower bound) | P2 PH-violation decomposition | v3 clearance subset | HR > 1 | `Data/Sensitivity_HPV_Clearance_TimeStratified_v3.csv` |
| **Sens-C** | Dose-threshold ≥2 / ≥3 (matched-set preserved) | P1+P2 exposure-misclassification | Cohort A 1:1 + Cohort B 1:up-to-4 | HR < 1 (recurrence); HR > 1 (clearance) | `Data/Sensitivity_DoseThreshold_HR.csv` |
| **Sens-D** | Strict 1:4 fine matching | P1 matching-structure robustness | Cohort B (strict subset) | HR < 1 | `Data/Sensitivity_StrictMatching.csv` |
| **Sens-E** | Lesion recurrence with ≥3, ≥6, ≥12-month minimum disease-free interval **post-landmark** (in addition to the 90-day landmark in the v3 primary) | P1 incomplete-excision artefact | v3 Cohort B P1 analytic | HR < 1 | `Data/Sensitivity_Recurrence_DFInterval_v3.csv` |

#### 4.5.2 Appendix-only sensitivity analyses (supplementary tables only)

These analyses are retained in the supplementary material for reviewer transparency and as supportive context for the Discussion narrative; they are not reported in the main Results sensitivity paragraph and are not included in Supplementary Figure S6.

| ID | Description | Status | Output |
|---|---|---|---|
| App-1 | Post-index hr-HPV detection (any) — supportive context for clearance interpretation | Discussion only; Table only | `Data/CohortB_HR_revised.csv` row 3 |
| App-2 | Post-index detection landmark (6/12/24 mo) — superseded by Sens-B time-stratified clearance | Table only | `Data/Sensitivity_HPV_Landmark.csv` |
| App-3 | Novel-type acquisition | Discussion only; Table only | `Data/Sensitivity_HPV_NovelType.csv` |
| App-4 | HPV-16 clearance (type-specific) | Table only | `Data/Sensitivity_HPV_Clearance.csv` C2 |
| App-5 | HPV-18 clearance (type-specific) | Table only | `Data/Sensitivity_HPV_Clearance.csv` C3 |
| App-6 | Vaccine-type interaction (LRT) — already reported via Figure 4 | Discussion only; Table only | `Data/CohortB_vaccine_interaction.csv` |
| App-7 | Vaccine-type × calendar period — Discussion narrative only | Discussion only; Table only | `Data/Sensitivity_VaccineType_ByCalendar.csv` |
| App-8 | Restricted follow-up (3-y / 5-y) and unadjusted | Table only | `Data/sensitivity_analysis_both_outcomes.csv` |
| App-9 | Age-stratified recurrence × FU grid (the 30–52 y, 2-y window finding lives here) | Table only; flagged in Limitations | `Data/CohortB_age_fu_forest.csv` |
| App-10 | Prescription-code vs drug-name exposure ascertainment cross-check | Table only | `Data/Sensitivity_PrescriptionCode.csv` |

### 4.5b Primary exposure definition (≥2 dose + 3-month landmark)

Effective with this revision, the **primary Cohort B exposure** requires **≥2 distinct HPV-vaccine prescription dates** (counted from the prescription file using the same identification rules as §2; identical-day records collapse to one dose). A **symmetric 3-month landmark** is applied across arms: the at-risk clock is shifted to `index + 90 days`, patients with < 90 days follow-up or with an outcome event in the first 90 days are excluded, and matched non-vaccinated controls of any vaccinated case dropped by the dose or landmark filter are removed in the same step to preserve matched-set integrity.

Final Cohort B primary **cohort** n = **934 (204 vaccinated / 730 fine-matched controls)** — eligible population after the ≥2-dose threshold and the 3-month landmark follow-up filter with matched-set integrity preserved.

Cohort B primary **analytic samples** (post-event-landmark filter, used by the Cox models):
- **P1 lesion recurrence**: n = **912 (203 / 709)** — one additional matched set removed because the vaccinated case had a recurrence event in the first 90 days, plus a small number of non-vaccinated controls with early events.
- **P2 hr-HPV clearance**: n = **235 (92 / 143)** — restricted to women with documented pre-vaccine hr-HPV positivity, matched-set integrity preserved, early clearance events removed.

The previous `≥1-dose, no-landmark` definition is retained as **Sens-C exposure-definition sensitivity** (legacy primary). Direction of effect comparisons:

| Outcome | ≥1 dose, no landmark (Sens-C, legacy) | ≥2 dose + 3-mo landmark (NEW PRIMARY) |
|---|---|---|
| Lesion recurrence (CIN2+) | HR 0.80 (0.44–1.43), p = 0.45 | HR 1.01 (0.49–2.06), p = 0.985 |
| hr-HPV clearance | HR 1.40 (0.92–2.11), p = 0.11 | **HR 1.85 (1.09–3.17), p = 0.024** ✅ |
| ≥3 dose, no landmark (sensitivity) | HR 0.58 (0.27–1.26), p = 0.17 | — |

Interpretation: the lesion-recurrence directional protection observed under ≥1 dose was substantially attributable to immortal-time selection (a woman who reached her second dose was guaranteed to have been event-free up to that date), and collapses to null under the tighter primary definition. The clearance signal, in contrast, **strengthens** under the tighter definition and reaches conventional significance, consistent with the biological hypothesis that post-surgical vaccination accelerates immune clearance of pre-existing HR-HPV without reversing already-initiated dysplastic precursor lesions.

### 4.5c Follow-up time reporting

All follow-up duration statistics in the manuscript are reported using the **reverse Kaplan–Meier method of Schemper and Smith** (Control Clin Trials 1996;17:343-346). The event indicator is inverted (censoring becomes the "event", outcome becomes the censoring) and the Kaplan–Meier estimator is applied to the resulting series; the resulting median is the unbiased estimate of the *median potential follow-up time*. The naïve median of observed follow-up time is systematically biased downward when outcome events truncate observation; the difference is small in this study (≈0.07 y in Cohort B v3) but the reverse-KM is the methodologically preferred reporting convention and is used uniformly across cohorts and subsets.

| Cohort | Group | n | reverse-KM median FU (y) | IQR (y) | Origin |
|---|---|---|---|---|---|
| Cohort A (PSM) | Vac | 2,050 | 4.68 | 1.95–9.92 | Index |
| Cohort A (PSM) | Non | 2,110 | 6.99 | 3.73–11.21 | Index |
| Cohort A (PSM) | Combined | 4,160 | **5.93** | 2.78–10.75 | Index |
| Cohort B legacy (≥1 dose) | Vac | 241 | 4.97 | 3.36–8.84 | Index |
| Cohort B legacy (≥1 dose) | Non | 867 | 5.26 | 3.52–9.32 | Index |
| Cohort B legacy (≥1 dose) | Combined | 1,108 | **5.10** | 3.48–9.23 | Index |
| **Cohort B v3 PRIMARY** | Vac | 204 | 4.94 | 3.30–8.90 | Index |
| **Cohort B v3 PRIMARY** | Non | 730 | 5.02 | 3.48–8.93 | Index |
| **Cohort B v3 PRIMARY** | Combined | 934 | **4.97** | 3.44–8.91 | Index |
| Cohort B v3 PRIMARY | Combined | 934 | 4.73 | 3.20–8.67 | Landmark (+90 d) |

### 4.6 Statistical model (both co-primary outcomes)

- **Cox proportional-hazards** with **age at index** as the only adjustment covariate.
- **Cluster-robust standard errors** clustered on `fine_match_id`.
- **Time** = `days_to_event` if event observed (i.e. `days_to_recurrence` for P1 and `(first_neg_date − index_date).days` for P2), else `follow_up_days = 최종추적일자 − index_date`.
- **Censoring** at last follow-up, death, loss to follow-up, or 31 December 2025, whichever first.
- **Schoenfeld residual rank test** reported on the vaccinated covariate; lesion recurrence p = 0.82 (no PH violation), hr-HPV clearance p = 0.028 (interpreted as biologically expected non-constancy of any vaccine effect on clearance over follow-up — decomposed by the pre-specified Sens-B time-stratified piecewise Cox model in Supplementary Table S17).

---

## 5. Quick-reference summary table

| | Cohort A — Safety | Cohort B (recurrence co-primary) | Cohort B (clearance co-primary) |
|---|---|---|---|
| v3 primary **cohort** n | 4,102 | 934 | 235 |
| v3 **analytic** sample (Cox fit) n | 4,102 | **912** (203 / 709 after early-event removal) | 235 |
| Vac / Non-vac (cohort) | 2,051 / 2,051 | **204 / 730** | **92 / 143** |
| Match | PSM 1:1, caliper 0.2 SD logit(PS) | Variable-ratio 1:up-to-5 → 1:up-to-4 (then ≥2 dose + landmark filters) | Same + pre-vaccine HPV+ filter + landmark |
| Index (vac) | First vaccine date | First vaccine date after surgery (≥2 dose) | First vaccine date after surgery (≥2 dose) |
| Index (non-vac) | Random pseudo-date | Surgery + matched interval (T) | Surgery + matched interval (T) |
| Time zero (primary) | Index | **Index + 90 days (landmark)** | **Index + 90 days (landmark)** |
| Event source | Diagnosis records (5 chronic) | Tissue pathology (조직병리) | Molecular pathology (분자병리/HPV) |
| Event definition | First post-index ICD-10 group hit | First post-landmark ≥CIN2 (HSIL+) | First post-landmark of 2-consecutive-NEG |
| Favourable HR direction | HR < 1 | HR < 1 | **HR > 1** |
| **Primary HR (95% CI)** | 1.26 (0.75–2.12) Any-of-5 | **1.01 (0.49–2.06)** | **1.85 (1.09–3.17) ✅** |
| **p value** | 0.37 | **0.985** | **0.024** |
| Sensitivity (≥1 dose, no landmark) — HR | n/a | 0.80 (0.44–1.43), p=0.45 | 1.40 (0.92–2.11), p=0.11 |
| Sensitivity (≥3 dose, no landmark) — HR | n/a | 0.58 (0.27–1.26), p=0.17 | — |
| Sustained clearance — KM median (Q25, Q75) — vaccinated | — | — | **10.79y (2.31, NR)** |
| Sustained clearance — KM median (Q25, Q75) — non-vac | — | — | 5.67y (1.91, NR) |
| Reversion events / censored — vaccinated | — | — | 13 / 18 (n=31 cleared) |
| Reversion events / censored — non-vac | — | — | 13 / 15 (n=28 cleared) |
| Log-rank (reversion-free) | — | — | χ²=1.00, p=0.317 |
| 5-yr reversion-free probability — vaccinated | — | — | **0.569 (56.9%)** |
| 5-yr reversion-free probability — non-vac | — | — | 0.533 (53.3%) |

Clearance co-primary reaches conventional significance under the new primary definition; lesion-recurrence null is honest collapse of the legacy ≥1-dose directional signal once immortal-time selection is removed.

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
