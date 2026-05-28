# Manuscript Outline — Academic Methods, Results, Tables/Figures

## 0. 제안 제목 (5 candidates)

연구 두 축 (long-term comorbidity safety + post-surgery vaccine efficacy)을 모두 담는 제목 후보:

1. ★ **"Long-term safety and post-surgical efficacy of HPV vaccination in women with cervical intraepithelial lesions: a prospective–retrospective cohort study using Korean clinical data warehouse"**
   - 두 축이 모두 명시되고, prospective→retrospective design hybrid도 드러남

2. **"HPV vaccination beyond cervical disease: long-term cardiovascular and metabolic outcomes and post-conization efficacy in a Korean cervical cohort"**
   - Beyond cervical disease 메타포로 안전성 측면 강조

3. **"Real-world evidence of HPV vaccination in Korean women with cervical intraepithelial lesions: long-term comorbidity surveillance and post-surgical recurrence prevention"**
   - Real-world evidence 키워드 부각

4. **"Dual-cohort analysis of HPV vaccination effects: chronic disease incidence and lesion recurrence in a Korean cervical cancer screening cohort"**
   - Dual-cohort design 강조

5. **"Post-surgery HPV vaccination and downstream outcomes: lesion recurrence, hr-HPV clearance (pre-vaccine HPV+ subset), and chronic comorbidities in a 15-year Korean prospective cohort"**
   - Long-term f/up 측면 부각

> **추천: #1** — 두 결과 축, prospective+retrospective hybrid, 지역 (Korean), 모집단 특성 (cervical intraepithelial lesions) 모두 반영. 부제로 "A Korean Cohort Study"는 옵션.

**한국어 제목 (국문초록용)**:
*"자궁경부 상피내 병변 환자에서 HPV 예방접종의 장기 안전성 및 수술 후 효능: 한국 임상자료저장소 기반 전향-후향 코호트 연구"*

---

## 1. METHODS — 학술논문용 구성안

### 1.1 Study design and data source
- 단일 의료기관 한국 HPV 코호트 (prospectively enrolled 2009–2024)
- 후향적 임상자료저장소(CDW) 추출로 추가 추적 자료 (진단·검사·처방·수술·기초임상정보·기저질환분류) 확보
- 본 연구는 prospective cohort + retrospective CDW augmentation에 의한 long-term follow-up 분석
- IRB 승인 (번호 TBD)

### 1.2 Source population (n=32,969)
- 자궁경부 진단·치료 기록을 보유한 모든 등록 환자
- 추적 종료: 2025-12-31, 사망일, 또는 자격상실일 중 가장 빠른 시점
- 모두 여성

### 1.3 Two analytic cohorts (rationale and selection)
| | **Cohort A — Safety analysis** | **Cohort B — Efficacy analysis** |
|---|---|---|
| 연구 질문 | Q1: HPV 백신이 장기적으로 만성질환 발생에 영향을 주는가? | Q2: 자궁경부 수술 후 HPV 백신이 병변 재발 및 HPV 재감염을 줄이는가? |
| 모집단 | 전체 코호트 (n=32,969) | 자궁경부 수술 시행자 (n=6,890) |
| 기본 매칭 | PSM 1:1 (age, BMI, SBP, DBP, smoking, residence) | 1:up-to-5 → fine 1:up-to-4 (수술방법 exact, 수술시점, age, BMI, 수술연도; variable-ratio greedy) |
| 최종 표본 (intermediate (≥1-dose, no-landmark sensitivity)) | 4,102 (vac 2,051 / ctl 2,051) | 1,108 (vac 241 / ctl 867) |
| **Primary (≥2 dose + 3-mo landmark)** | 4,102 (unchanged) | **912 (vac 203 / ctl 709)** |
| Index date (vac) | 첫 백신 접종일 | 첫 백신 접종일 |
| Index date (ctl) | 매칭된 접종군 백신일 (pseudo) | 비접종군 수술일 + 매칭 접종군의 수술-접종 간격(T) |
| Primary outcome | 5대 만성질환 발생, MCE, Any-of-5 | 병변 재발 (CIN2 이상 / HSIL+), hr-HPV clearance |
| Secondary | (없음) | 백신 종류별 subgroup (Gardasil9 / Cervarix / Gardasil) |

### 1.4 Exposure ascertainment
- HPV 백신 접종 식별: 처방정보에서 "Gardasil", "Gardasil 9", "Cervarix", "HPV vaccine", 가다실, 서바릭스 키워드
- 첫 접종일을 index date로 정의

### 1.5 Outcome definitions

**Cohort A (Safety)**
- 5개 사전 정의된 만성질환 (CDW의 진단정보 분류 라벨 기반):
  1. 협심증/심근경색 (Angina/MI; ICD-10 I20–I25)
  2. 고혈압 (Hypertension; I10–I15)
  3. 당뇨 (Diabetes; E11–E14)
  4. 뇌출혈/뇌경색 (Stroke; I60–I69)
  5. 폐색전증 (Pulmonary embolism; I26)
- **Composite endpoints**:
  - **MCE (Major Cardiovascular Events)**: First occurrence of MI, stroke, or PE
  - **Any-of-5**: First occurrence of any of the 5 comorbidities
- Prevalent (baseline 진단 보유) 환자는 해당 outcome 분석에서 제외

**Cohort B (Efficacy)**
- **Primary**: 병변 재발 — index date 이후 첫 조직검사에서 ≥CIN2 (HSIL+) 또는 invasive cervical cancer 진단
- **Primary**: 고위험 HPV 재감염 — index date 이후 14개 고위험 유형 (16, 18, 31, 33, 35, 39, 45, 51, 52, 56, 58, 59, 66, 68) 양성 전환

### 1.6 Matching procedures

**Cohort A — Propensity score matching (1:1)**
- Logistic regression PS model with covariates: age at index, BMI (mean imputation + missing indicator), SBP (idem), DBP (idem), smoking status (Never/Former/Current/Unknown reference), residence in Seoul
- 1:1 nearest-neighbor matching on logit(PS), caliper 0.2 × SD(logit PS), without replacement (Austin 2011)
- Pseudo index date for non-vaccinated: random sample from vaccinated patients' first-vaccine-date distribution (seed=42)
- Eligibility: alive at index, ≥1 day of follow-up

**Cohort B — Variable-ratio matching (1:up-to-4)**
- Algorithm note: Both steps use **greedy nearest matching without replacement**; the requested ratio is the *maximum* number of controls per vaccinated case. If fewer eligible controls exist (rare cells, depleted pool), the case is matched with whatever controls remain.
- Step 1 — 1:up-to-5 initial match: 수술방법 (exact: conization vs hysterectomy), 수술시점 (±1 yr), age at surgery (±5 yr) → 411 vaccinated and 1,815 non-vaccinated (mean ratio 4.42; 256 of 411 cases reached the maximum of 5).
- Step 2 — Index date filtering: index ≤ 2020-12-31 (≥5 yr follow-up), ≥2 follow-up records → 411 / 1,797 (18 controls excluded).
- Step 3 — Fine matching to 1:up-to-4: index age (±5 y), index BMI (±3 kg/m²), surgery year (±1 y) → 241 vaccinated and 867 non-vaccinated (mean ratio 3.60; 193 of 241 cases reached the maximum of 4). **Legacy intermediate cohort.**
- Step 4 (PRIMARY) — ≥2-dose + 3-month landmark filter with matched-set integrity preserved: 36 vaccinated cases failing the dose threshold and their attached non-vaccinated participants dropped; 5 patients lost to the 3-month landmark FU filter; final  primary cohort **n = 912 (203 vaccinated / 709 non-vaccinated)**.
- Pseudo index date for non-vaccinated: surgery date + matched 접종군의 수술-접종 간격 T

### 1.7 Covariate balance assessment
- Standardized mean difference (SMD); |SMD|<0.10 = well balanced
- Love plots before vs after matching

### 1.8 Statistical analysis

**Cohort A**
- Cumulative incidence functions: Aalen-Johansen estimator (death as competing event)
- Cause-specific hazard ratios: Cox proportional-hazards (death = censored), robust SE
- Subdistribution hazard ratios (Fine-Gray): time-varying Cox on Geskus (2011) IPCW-reweighted dataset, robust SE

**Cohort B**
- Kaplan-Meier estimator + log-rank test
- Cox proportional-hazards model adjusting for residual age imbalance, robust SE
- Subgroup analysis by vaccine type

**Software**
- Python 3.14, lifelines 0.30, scikit-learn, scipy, pandas, numpy
- All matching seeds = 42 for reproducibility

### 1.9 Sensitivity analyses (Cohort B)
- Restricted follow-up (3-yr, 5-yr)
- As-treated analysis
- Adjusted vs unadjusted Cox

---

## 2. RESULTS — 학술논문용 구성안

### 2.1 Cohort selection and characteristics
**Figure 1.** Cohort selection flow diagram (single source → two analytic cohorts).

**Table 1.** Baseline characteristics of two analytic cohorts (post-matching), with absolute SMDs. Pre-matching version → Supplementary Table S1.

### 2.2 Cohort A — Vaccination and chronic disease incidence (Q1)

#### 2.2.1 Person-time and event rates
- Total person-years
- Crude incidence per 1,000 person-years for each comorbidity, MCE, Any-of-5

#### 2.2.2 Cumulative incidence
**Figure 2.** Aalen-Johansen cumulative incidence curves (vaccinated vs non-vaccinated) for:
- (a) Any-of-5 composite
- (b) MCE composite (MI/Stroke/PE)
- (c)–(g) Individual comorbidities

#### 2.2.3 Hazard ratios
**Table 2.** Cause-specific HRs and Fine-Gray subdistribution sHRs with 95% CI for each endpoint.
- Order: Any-of-5, MCE, then individual outcomes

**Forest plot** within Figure 2 (panel h): cause-specific vs Fine-Gray HRs side-by-side.

### 2.3 Cohort B — Post-surgical vaccine efficacy (Q2)

#### 2.3.1 Outcome rates
**Table 3.** Vaccinated vs non-vaccinated event rates and Cox-adjusted hazard ratios for:
- Lesion recurrence (≥CIN2 / HSIL+ or invasive cervical cancer)
- High-risk hr-HPV clearance (pre-vaccine HPV+ subset)

#### 2.3.2 Survival curves
**Figure 3 ().** Cumulative incidence curves (1 − Kaplan-Meier) anchored at the 3-month landmark for (a) lesion recurrence in P1 analytic n=912 and (b) hr-HPV clearance in P2 analytic n=233, with HR/CI/p annotation and number-at-risk table.

#### 2.3.3 Combined subgroup forest
**Figure 4 ().** JAMA-style combined table-with-forest plot of HRs by age (<40 / 40–49 / ≥50) and vaccine type (Gardasil 9 / Cervarix / quadrivalent Gardasil) for both co-primary outcomes. Replaces legacy separate Figure 4 (vaccine type only) and Figure 5 (age subgroups).

#### 2.3.4 Sensitivity analyses
**Supplementary Figure S6 ().** Five-panel summary forest plot — Sens-A (single-neg vs 2-cons-neg clearance), Sens-B (time-stratified clearance with 0–6mo signal), Sens-C (dose threshold), Sens-D (strict 1:4 matching), Sens-E (recurrence DFI 0/3/6/12mo post-index).

---

## 3. PROPOSED TABLES/FIGURES

### 3.1 Main Tables

| # | 제목 | 핵심 내용 | 산출 파일 |
|---|---|---|---|
| **Table 1** | Baseline characteristics of analytic cohorts (post-matching) | Demographics, BMI/BP, smoking, comorbidities, follow-up — Cohort A + Cohort B columns | `Table1_BaselineCharacteristics_unified.docx` (post 행만 추출) |
| **Table 2** | Hazard ratios for chronic comorbidities (Cohort A) | Cause-specific HR + Fine-Gray sHR, events/n, p — Any-of-5 / MCE / individual 5 | `cohort_a_psm_hr_results.csv` |
| **Table 3** | Vaccine effectiveness on lesion recurrence and hr-HPV clearance (pre-vaccine HPV+ subset) (Cohort B) | n events, HR (95% CI), p, by overall and adjusted | (기존 `vaccine_type_analysis.csv` + `final_matched_summary.csv` 기반) |

### 3.2 Main Figures

| # | 제목 | 내용 | 산출 파일 |
|---|---|---|---|
| **Figure 1** | Cohort selection flow diagram | CONSORT-style. Single source → two analytic cohorts | Mermaid (`Methods_CohortSelection.md`) → export to PNG |
| **Figure 2** | Cohort A: cumulative incidence and hazard ratios | 9-panel: Any-of-5, MCE, 5 individual, forest plot | `cohort_a_psm_cif_hr.png` |
| **Figure 3** | Cohort B: Kaplan-Meier survival curves | (a) recurrence, (b) hr-HPV clearance (pre-vaccine HPV+ subset) + risk table | `figure1_kaplan_meier.png` |
| **Figure 4** | Cohort B: Forest plot by vaccine type | Subgroup HRs (Gardasil9/Cervarix/Gardasil) for both outcomes | `figure5_vaccine_forest.png` |

### 3.3 Supplementary Tables

| # | 제목 | 내용 | 산출 파일 |
|---|---|---|---|
| **S1** | Pre-matching baseline characteristics (full) | Cohort A pre vs Cohort A post; Cohort B pre vs post — full variable list | `Table1_BaselineCharacteristics_unified.docx` |
| **S2** | Propensity score model coefficients (Cohort A) | LogReg coefs, ORs, CI for each PS covariate | (스크립트 출력에서 추출) |
| **S3** | Sensitivity analyses (Cohort B) | Restricted follow-up, as-treated, adjusted/unadjusted; HR and 95% CI | `sensitivity_analysis_results.csv`, `sensitivity_age_cutoff.csv` |
| **S4** | Subgroup analysis by age (Cohort B) | Age strata × follow-up window 조합별 HR | `subgroup_analysis_comprehensive.csv` |
| **S5** | Number-at-risk tables for KM curves | Time × group | (KM plot 보조표) |
| **S6** | Per-vaccine-type detailed results | Gardasil9 / Cervarix / Gardasil — events, HR, CI for both outcomes | `vaccine_type_analysis.csv` |

### 3.4 Supplementary Figures

| # | 제목 | 내용 | 산출 파일 |
|---|---|---|---|
| **S1** | Love plot — Cohort A PSM | |SMD| pre vs post for all PS covariates | `cohort_a_psm_loveplot.png` |
| **S2** | Love plot — Cohort B fine matching | |SMD| pre vs post for matching variables | `love_plot.png` |
| **S3** | Propensity score distribution | Density plot of PS by vaccinated/non-vaccinated, before/after matching | (신규 생성 필요) |
| **S4** | Cohort B subgroup forest by age (sensitivity) | HRs across age strata × follow-up window | `figure3_subgroup_km.png` 또는 신규 |
| **S5** | Cohort A — Smoothed hazard ratio over time | Cumulative HR estimate for Any-of-5 over follow-up years | (옵션, 신규) |

---

## 4. Manuscript flow proposal (목차)

```
Title page
Abstract (250 words, structured: Background / Methods / Results / Conclusions)
Keywords: HPV vaccine; cervical intraepithelial neoplasia; long-term safety; lesion recurrence; cohort study; propensity score matching

1. Introduction
   - HPV burden, vaccine landscape
   - Gap: long-term safety beyond cervical lesions, post-surgery efficacy in Korean women
   - Aims (Q1, Q2)

2. Methods (위 §1 참조)

3. Results (위 §2 참조)

4. Discussion
   - Principal findings
     · Q1: 백신 접종 후 5대 만성질환 발생 차이 없음 (NS for all individual + composites)
     · Q2: 전체 NS, but Gardasil(4가) HPV 재감염에서 유의 / 30–52세 2년 추적 병변 재발 유의
   - Comparison with prior literature (FUTURE I/II, KEN-SHE, real-world Korean data)
   - Mechanism: PSM 후 잔여 healthy-vaccinee bias 고려
   - Strengths: dual-cohort design, CDW long-term f/up, competing-risk framework
   - Limitations: single center, 작은 사건 수 (특히 CV outcome), 매칭 불가능한 unmeasured confounders, female-only

5. Conclusions

References

Tables 1–3
Figures 1–4
Supplementary Tables S1–S6
Supplementary Figures S1–S5
```

---

## 5. 다음 단계 제안

1. **Cohort B 결과 산출 동기화**: 기존 fine-matched cohort에서 KM/Cox/forest 그림과 표를 main figure 규격으로 재출력 (label/font 통일)
2. **Person-time 및 incidence rate (Cohort A)**: Crude IR per 1,000 person-years 계산 → Table 2 보강
3. **Supplementary Figure S3 (PS distribution)**: 신규 작성
4. **Abstract 250-word 초안**: Methods/Results 골격 확정 후 작성
5. **Reference list 준비**: HPV 백신 효능 (FUTURE I/II), 백신 안전성 메타분석, Fine-Gray 방법론, propensity score matching 가이드 등

원하시는 항목부터 진행해드릴게요.
