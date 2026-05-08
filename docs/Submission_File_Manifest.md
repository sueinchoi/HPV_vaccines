# Manuscript Submission — File Manifest and Naming Guide

**Manuscript**: Long-term safety and post-surgical efficacy of HPV vaccination in women with cervical intraepithelial lesions: a prospective–retrospective cohort study using Korean clinical data warehouse
**Last updated**: 2026-05-02
**Repository root**: `/Users/sueinchoi/GitHub/HPV_vaccines/`

---

## 0. Submission folder layout

When preparing the submission, copy the source files from `Data/` into a clean `submission/` folder using the standardised filenames in column **Submit as**. Convert to the format requested by the journal (see §4).

```
submission/
├── manuscript.docx                    # main text
├── tables/
│   ├── Table1_BaselineCharacteristics.docx
│   ├── Table2_CohortA_HazardRatios.docx
│   └── Table3_CohortB_HazardRatios.docx
├── figures/
│   ├── Figure1_CohortSelection.png
│   ├── Figure2_CohortA_CIF_HR.png
│   ├── Figure3_CohortB_KaplanMeier.png
│   ├── Figure4_CohortB_VaccineType.png
│   └── Figure5_CohortB_AgeFollowUp.png
└── supplementary/
    ├── SupTableS1_Baseline_PreMatching.docx
    ├── SupTableS2_PropensityScoreCoefficients.docx
    ├── SupTableS3_Sensitivity.csv
    ├── SupTableS4_AgeFollowUpSubgroup.docx
    ├── SupTableS5_NumberAtRisk.docx
    ├── SupTableS6_VaccineTypeDetails.csv
    ├── SupTableS7_ClusterRobustHRs.docx
    ├── SupTableS8_PseudoIndexSensitivity.csv
    ├── SupFigS1_LovePlot_CohortA.png
    ├── SupFigS2_LovePlot_CohortB.png
    ├── SupFigS3_PropensityScoreDensity.png
    ├── SupFigS4_AgeSubgroupKM.png
    ├── SupFigS5a_PHCheck_CohortA_AnyOf5.png
    ├── SupFigS5b_PHCheck_CohortA_Diabetes.png
    ├── SupFigS6a_PHCheck_CohortB_Recurrence.png
    └── SupFigS6b_PHCheck_CohortB_HPVReinfection.png
```

---

## 1. Main Tables

| # | Title | Source file (in `Data/`) | Submit as | Format |
|---|---|---|---|---|
| **Table 1** | Baseline characteristics of analytic cohorts (post-matching). Demographics, anthropometry, vital signs, smoking, comorbidities, and follow-up for Cohort A and Cohort B with absolute standardised mean differences. | `Table1_BaselineCharacteristics_unified.docx` (post-matching blocks (b) and (d) only — keep page breaks; remove pre-matching blocks (a)·(c) before submission) | `Table1_BaselineCharacteristics.docx` | DOCX |
| **Table 2** | Cohort A — Cluster-robust cause-specific and Fine–Gray subdistribution hazard ratios with person-years and incidence rates for the Any-of-5 composite, MCE composite, and five individual chronic conditions. | `Methodology_Revisions.docx` (Table R1 only) **OR** rebuild from `CohortA_HR_revised.csv` | `Table2_CohortA_HazardRatios.docx` | DOCX |
| **Table 3** | Cohort B — Age-adjusted Cox cluster-robust hazard ratios with person-years and incidence rates for biopsy-confirmed lesion recurrence and high-risk HPV reinfection. | `Methodology_Revisions.docx` (Table R2 only) **OR** rebuild from `CohortB_HR_revised.csv` (preferred — the standalone `Table3_CohortB_HR.docx` is from the older non-clustered model) | `Table3_CohortB_HazardRatios.docx` | DOCX |

> **Note for Table 3**: The earlier `Data/Table3_CohortB_HR.docx` was generated before cluster-robust SE was applied. Use the values in `CohortB_HR_revised.csv` (HR 0.76 for recurrence, 0.97 for HPV reinfection) — these match the manuscript text. Bonferroni-adjusted columns and 97.5% CIs were dropped per the final analysis plan and should not appear in the table.

---

## 2. Main Figures

| # | Title | Source file (in `Data/`) | Submit as | Format / Resolution |
|---|---|---|---|---|
| **Figure 1** | Cohort selection flow diagram — single source population (N = 32,969) → Cohort A (whole-cohort safety, n = 4,102) and Cohort B (post-surgical efficacy, n = 1,108). | `Figure1_cohort_selection.png` | `Figure1_CohortSelection.png` | PNG, 200 dpi |
| **Figure 2** | Cohort A — Aalen–Johansen cumulative incidence functions and cause-specific / Fine–Gray hazard ratio forest plot for Any-of-5, MCE, and five individual comorbidities. | `cohort_a_psm_cif_hr.png` | `Figure2_CohortA_CIF_HR.png` | PNG, 150 dpi |
| **Figure 3** | Cohort B — Kaplan–Meier curves for biopsy-confirmed lesion recurrence and high-risk HPV reinfection. | `figure1_kaplan_meier.png` | `Figure3_CohortB_KaplanMeier.png` | PNG (existing) |
| **Figure 4** | Cohort B — Forest plot of hazard ratios by vaccine type (Gardasil 9, Cervarix, Gardasil) for both primary outcomes. | `figure5_vaccine_forest.png` | `Figure4_CohortB_VaccineType.png` | PNG (existing) |
| **Figure 5** | Cohort B — Forest plot of age stratum × follow-up window (1 / 2 / 4 yr / Full) subgroup analysis for lesion recurrence. | `CohortB_age_fu_forest.png` | `Figure5_CohortB_AgeFollowUp.png` | PNG, 200 dpi |

---

## 3. Supplementary Materials

### 3.1 Supplementary Tables

| # | Title | Source file (in `Data/`) | Submit as | Format |
|---|---|---|---|---|
| **S1** | Pre-matching baseline characteristics (full variable list as Table 1, before matching for both cohorts). | `Table1_BaselineCharacteristics_unified.docx` (pre-matching blocks (a) and (c) only) | `SupTableS1_Baseline_PreMatching.docx` | DOCX |
| **S2** | Cohort A propensity score model — logistic regression coefficients (original-scale and standardised), odds ratios, and intercept. | `SupTableS2_ps_coefficients.docx` | `SupTableS2_PropensityScoreCoefficients.docx` | DOCX |
| **S3** | Cohort B sensitivity analyses — restricted follow-up (3 yr, 5 yr) and adjusted vs. unadjusted Cox models. | `sensitivity_analysis_results.csv` | `SupTableS3_Sensitivity.csv` | CSV (or convert to DOCX) |
| **S4** | Cohort B subgroup analysis — pre-specified age strata (<40, 40–49, ≥50, plus 30–52 post-hoc) × follow-up windows (1, 2, 4 yr, full) for lesion recurrence. | `SupTableS4_revised_age_fu_forest.docx` | `SupTableS4_AgeFollowUpSubgroup.docx` | DOCX |
| **S5** | Number-at-risk tables for Kaplan–Meier and Aalen–Johansen curves by group at yearly intervals (0–8 yr). | `SupTableS5_number_at_risk.docx` | `SupTableS5_NumberAtRisk.docx` | DOCX |
| **S6** | Per-vaccine-type detailed results — pairwise subgroup hazard ratios and single-model interaction-derived hazard ratios with the LRT for vaccine-type heterogeneity. | `vaccine_type_analysis.csv` + `CohortB_vaccine_interaction.csv` (concatenate) | `SupTableS6_VaccineTypeDetails.csv` | CSV (or convert to DOCX) |
| **S7** | Cluster-robust hazard ratios with person-years, incidence rates, and Schoenfeld residual p-values for both cohorts. | `Methodology_Revisions.docx` (Tables R1 + R2) | `SupTableS7_ClusterRobustHRs.docx` | DOCX |
| **S8** | Pseudo-index assignment sensitivity analysis for Cohort A (Any-of-5 endpoint) — random sample, calendar-year-matched, and risk-set sampling strategies. | `CohortA_pseudoindex_sensitivity.csv` | `SupTableS8_PseudoIndexSensitivity.csv` | CSV |

### 3.2 Supplementary Figures

| # | Title | Source file (in `Data/`) | Submit as | Format / Resolution |
|---|---|---|---|---|
| **S1** | Love plot — covariate balance before and after 1:1 propensity score matching (Cohort A). | `SupFigS1_loveplot_cohortA.png` | `SupFigS1_LovePlot_CohortA.png` | PNG, 200 dpi |
| **S2** | Love plot — covariate balance before and after fine variable-ratio (1:up-to-4) matching (Cohort B). | `SupFigS2_loveplot_cohortB.png` | `SupFigS2_LovePlot_CohortB.png` | PNG, 200 dpi |
| **S3** | Propensity score density distributions before and after matching (Cohort A). | `SupFigS3_ps_density.png` | `SupFigS3_PropensityScoreDensity.png` | PNG, 200 dpi |
| **S4** | Subgroup forest plot by age strata for Cohort B (legacy single-figure view; superseded by Figure 5 — retain as supplementary). | `figure3_subgroup_km.png` | `SupFigS4_AgeSubgroupKM.png` | PNG (existing) |
| **S5a** | Schoenfeld residual diagnostic plot — Cohort A, Any-of-5 endpoint. | `PH_check_A_0.png` | `SupFigS5a_PHCheck_CohortA_AnyOf5.png` | PNG, 130 dpi |
| **S5b** | Schoenfeld residual diagnostic plot — Cohort A, Diabetes endpoint. | `PH_check_A_1.png` | `SupFigS5b_PHCheck_CohortA_Diabetes.png` | PNG, 130 dpi |
| **S6a** | Schoenfeld residual diagnostic plot — Cohort B, lesion recurrence. | `PH_check_B_has_recurrence.png` | `SupFigS6a_PHCheck_CohortB_Recurrence.png` | PNG, 130 dpi |
| **S6b** | Schoenfeld residual diagnostic plot — Cohort B, HPV reinfection. | `PH_check_B_has_hpv_infection.png` | `SupFigS6b_PHCheck_CohortB_HPVReinfection.png` | PNG, 130 dpi |

---

## 4. Format conversion checklist

Most journals require:

| Asset type | Preferred format | Acceptable | Notes |
|---|---|---|---|
| Main figures | TIFF (LZW) ≥ 300 dpi or EPS/PDF (vector) | High-res PNG | Keep colour mode RGB unless print journal asks CMYK. Embed all fonts. |
| Main tables | DOCX | RTF / typeset in manuscript | Avoid landscape unless necessary; use journal's table style. |
| Supplementary figures | PNG/PDF | TIFF | Compress if > 5 MB. |
| Supplementary tables | DOCX, CSV, or XLSX | PDF | Single supplementary document (`Supplementary_Material.docx`) often preferred — combine with explicit captions. |

**Conversion commands (macOS / Linux)**

```bash
# PNG → TIFF (300 dpi)
sips -s format tiff --setProperty dpiHeight 300 --setProperty dpiWidth 300 figure.png --out figure.tiff

# CSV → DOCX (preserve simple table layout)
pandoc -f csv -t docx file.csv -o file.docx

# Combine supplementary into a single docx
pandoc SupTableS*.md SupFigS*.md -o Supplementary_Material.docx
```

---

## 5. Files NOT used in the final submission

The following analytic outputs in `Data/` were intermediate or superseded; **do not include** them in the submission package:

- `Table1_BaselineCharacteristics.docx` (early version — superseded by `_unified.docx`)
- `Table1_BaselineCharacteristics.csv` (early version)
- `Table3_CohortB_HR.docx` / `.csv` (pre-cluster-robust — superseded by `Methodology_Revisions.docx` Table R2 / `CohortB_HR_revised.csv`)
- `cohort_a_psm_loveplot.png` (early Cohort A love plot — superseded by `SupFigS1_loveplot_cohortA.png`)
- `love_plot.png` (early Cohort B love plot with Korean labels — superseded by `SupFigS2_loveplot_cohortB.png`)
- `figure2_forest_plot.png` (legacy single-outcome forest — superseded by Figure 4 layout)
- `figure4_baseline.png` (legacy baseline figure — superseded by Table 1)
- `figure6_vaccine_group_forest.png` (legacy duplicate of Figure 4)
- `subgroup_fine_tuning.csv` (intermediate diagnostic output)
- `sensitivity_age_cutoff.csv` (full ≈6,900-row grid search; cite in manuscript footnote but do not attach as supplementary table — the pre-specified subset is in S4)
- `outcomes_summary.csv`, `patient_outcomes_summary.csv`, `cohort_outcomes.csv` (intermediate outputs)
- `pathology_sample.csv`, `한국 HPV 코호트 자료를 이용한 자_*.csv` (raw source data — do not share)

---

## 6. Manuscript text — exact in-text references

When citing files in the manuscript text, use the **figure/table numbers only** (e.g., "Figure 2", "Supplementary Table S4"). The submission package's standardised filenames in column **Submit as** above ensure that the editorial system can match each callout to its file.

The current `docs/Manuscript_Draft.md` references are already consistent with the numbering above. Before final submission, run:

```bash
grep -nE "(Figure|Table)\s*[0-9SR]" docs/Manuscript_Draft.md | sort -u
```

to verify every callout has a corresponding entry in this manifest.

---

## 7. Reproducibility chain

For each manuscript asset, the script that produces it is recorded below for the journal's reproducibility statement and for any future revisions.

| Asset | Script |
|---|---|
| Table 1, Sup Table S1 | `scripts/baseline_table1_unified.py` |
| Table 2, Sup Table S7, Figure 2 | `scripts/cohort_a_psm_hr_cif.py` and `scripts/methodology_revision.py` |
| Table 3, Sup Table S7 | `scripts/methodology_revision.py` |
| Sup Table S2 (PS coefficients) | `scripts/cohort_a_psm.py` and `scripts/generate_missing_artifacts.py` |
| Sup Table S3 | `scripts/sensitivity_analysis.py` (and v2/v3 variants) |
| Sup Table S4, Figure 5 | `scripts/age_fu_forest.py` |
| Sup Table S5 | `scripts/generate_missing_artifacts.py` |
| Sup Table S6 (interaction) | `scripts/methodology_revision.py` |
| Sup Table S8 (pseudo-index sensitivity) | `scripts/methodology_revision.py` |
| Sup Fig S1, S2 | `scripts/regenerate_love_plots.py` |
| Sup Fig S3 | `scripts/generate_missing_artifacts.py` |
| Sup Fig S5a/b, S6a/b (PH plots) | `scripts/methodology_revision.py` |
| Figure 1 | `scripts/generate_missing_artifacts.py` |
| Figure 3, 4, Sup Fig S4 | `scripts/analyze_cohort.py`, `scripts/vaccine_type_analysis.py` |

All scripts use `random_seed = 42` for reproducibility.

---

## 8. Pre-submission checklist

- [ ] All figures saved at journal-required resolution and format
- [ ] Filenames match the **Submit as** column (case-sensitive)
- [ ] Each figure caption ≤ 50 words; each table footnote present
- [ ] Bonferroni-adjusted columns and 97.5% CIs removed from Table 3
- [ ] Cohort A median follow-up updated where needed (currently reported as approximately 5.9 years mean for Cohort A and 4.88 yr median for Cohort B vaccinated, 5.02 yr for controls)
- [ ] Korean-labelled assets (`love_plot.png`) replaced with English-only counterparts
- [ ] Old Bonferroni columns removed from any inherited tables
- [ ] PH check supplementary figures attached
- [ ] Reproducibility appendix lists Python 3.14.3, lifelines 0.30.3, scikit-learn 1.8.0, scipy 1.17.1, pandas 2.3.3, numpy 2.4.2
- [ ] IRB number filled in (currently placeholder)
- [ ] Author affiliations and ORCIDs added
- [ ] Conflict-of-interest and funding statements complete
