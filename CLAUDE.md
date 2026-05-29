# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

이 저장소는 한국 임상데이터웨어하우스(CDW)를 이용한 HPV 백신 효과·안전성에 대한 후향적 매칭 코호트 연구의 분석 코드와 원고/공급자료 산출 파이프라인이다. 목표 산출물은 동료 심사용 manuscript (`docs/HPV_manuscript.docx`)와 부속 자료 (`docs/HPV_tables_figures.docx`, `docs/HPV_supplementary.docx`).

## 권위 있는 사양 문서

코호트 정의·index date·outcome·매칭 규칙·민감도 분석 ID에 대한 single source of truth는 **`docs/Analysis_Specifications.md`** 이다. 분석 변경 시 이 파일을 먼저 확인·갱신하고, README.md (초기 연구계획서)와 본 CLAUDE.md는 보조 자료로 취급한다. 제출용 산출물 명명·매핑은 `docs/Submission_File_Manifest.md` 참고.

## 두 개의 코호트

| | Cohort A — 장기 안전성 | Cohort B — 수술 후 효과 (**primary**) |
|---|---|---|
| 모집단 | 전체 cohort (N = 32,969) | 자궁경부 수술(원추절제술/자궁절제술) 환자 (N_pool = 6,890) |
| 노출 (primary) | **≥2 dose** HPV 백신 접종 (symmetric) | **수술 이후 ≥2 dose** HPV 백신 접종 |
| Time zero (primary) | **Index + 90 days (3-mo landmark, symmetric)** | **Index + 90 days (3-mo landmark, symmetric)** |
| Index (vac) | 첫 백신일 | 수술 이후 첫 백신일 |
| Index (non-vac) | 접종군 백신일 분포에서 random pseudo-date (seed=42) | 수술일 + 매칭 접종군의 (수술→접종 간격 T) |
| 매칭 | PSM 1:1, caliper 0.2 × SD logit(PS); 변수: 연령, BMI, SBP, DBP, 흡연, 서울 거주 | Step1: 수술방법(exact)/수술년(±1y)/수술시 연령(±5y) 1:up-to-5 → Step2: index ≤2020-12-31 + 추적≥2건 → Step3: index 연령(±5y)/BMI(±3 kg/m²)/수술년(±1y) 1:up-to-4 (BMI 결측 시 완화) → **Step4: ≥2 dose + 3-mo landmark filter w/ matched-set integrity** |
| 최종 N (primary cohort) | **2,776 (1,396/1,380)** (1:1 PSM 4,106 → ≥2 dose + 3-mo landmark) | **912 (203/709)** primary cohort (at-risk at landmark); P2 clearance subset **233 (92/141)** (pre-vaccine hr-HPV+ matched-set-integrity subset) |
| Primary HR | Any-of-5 **1.28 (0.66–2.48), p=0.47** | P1 **1.01 (0.49–2.06), p=0.99** / P2 **1.82 (1.07–3.11), p=0.027 ✅** |
| Outcome | 5개 만성질환 (협심증/MI, HTN, DM, 뇌졸중, PE) + Any-of-5 + MCE; **첫 post-index ICD-10 hit** | P1: 병변 재발(≥CIN2/HSIL+/암; **CIN2** 임계임을 주의); P2: hr-HPV clearance (post-index 분자병리 2건 연속 음성 중 첫 음성일자) |
| 효과 방향 | **HR < 1 유리** | P1: **HR < 1 유리** / P2: **HR > 1 유리** (clearance) |
| Primary HR (95% CI) | 1.26 (0.75–2.12) Any-of-5 | P1 **1.01 (0.49–2.06), p=0.99 (null collapse)** / P2 **1.82 (1.07–3.11), p=0.027 ✅** |
| Sensitivity (≥1 dose, no landmark) | n/a | P1 0.80 (0.44–1.43), p=0.45 / P2 1.40 (0.92–2.11), p=0.11 |
| Sustained clearance (KM median, Q25/Q75) | — | vac **10.79y (2.31, NR)** / non-vac 5.67y (1.91, NR); log-rank p=0.317 (reversion 13/31 vs 13/28); 5y reversion-free P 0.569 vs 0.533 |

⚠ Cohort B의 hr-HPV baseline은 **pre-vaccine** (records with `실시일자 < index_date`)으로 두 군 공통 — pre-surgery가 **아니다**. Matched-set 무결성: vaccinated case에 baseline HPV+ 기록이 없으면 그 `fine_match_id` 전체를 drop.

⚠ **primary 정의 (현행)**: ≥2 dose + 3-mo landmark + matched-set integrity (vaccinated case가 dose/landmark/pre-landmark-event 중 하나라도 실패하면 해당 `fine_match_id` 전체 drop) → **최종 912명 (203 vac / 709 unvac)**. 이전 ≥1 dose, no-landmark 분석은 Sens-C (`Analysis_Specifications.md §4.5b`)로 강등. 본문의 lesion-recurrence "null collapse" (Sens-C HR 0.80 → primary HR 1.01)는 immortal-time selection을 정직하게 보고하는 형태로 Limitations에 명시.

## 통계 규약 (`Analysis_Specifications.md` §4.6)

- Cox PH, 보정공변량 = **age at index만** (PSM/fine-matching이 잔여 교란을 잡았다는 가정).
- **Cluster-robust SE**: Cohort A는 `pair_id`, Cohort B는 `fine_match_id` 단위 클러스터.
- Time = `days_to_event` (이벤트 발생 시) / `follow_up_days = 최종추적일자 − index_date` (검열).
- 검열: 마지막 추적, 사망, 자격상실, 2025-12-31 중 최초.
- 사망은 competing event (Aalen–Johansen CIF, Fine–Gray subdistribution).
- 모든 매칭/sampling: `random seed = 42`, **without replacement**, variable-ratio는 *상한* (가용 통제군 < 상한이면 가용한 만큼만; `1:up-to-N` 표기).

## 노출 정의 (HPV 백신, `Analysis_Specifications.md` §2)

처방 레코드가 다음 중 하나라도 만족:
- `처방명` ∼ `/Gardasil|Cervarix|HPV vaccine/i`
- `처방한글명` ∼ `/가다실|서바릭스/`
- `처방코드`가 `DV-9HPF` (Gardasil 9) / `DV-HPF` (Gardasil 4가) / `DV-JHP` (Cervarix) 로 시작 (`-FR` 무료, `-FJ` 직원/가족, legacy `DV-HPJ` 포함)

`first_vaccine_date` = 자격 충족 처방의 최초 `처방일자`. 다종 접종(33명) 시 vaccine-type은 **첫 dose** 기준.

## 핵심 데이터 파일 (`Data/`)

원본 (cp949, PHI, gitignored) — 파일명 그대로 한글 컬럼 사용:
- `한국 HPV 코호트 자료를 이용한 자_코호트.csv` — 모집단 (N=32,969)
- `한국 HPV 코호트 자료를 이용한 자_수술처방_수술종류구분완료.csv` — 수동 분류 (`수술 종류`: 1=원추절제, 3=자궁절제, `제외`)
- `한국 HPV 코호트 자료를 이용한 자_처방정보.csv` — 백신 식별용
- `한국 HPV 코호트 자료를 이용한 자_진단정보_기저질환추가_unlocked.xlsx` — 사전 분류된 5개 만성질환 tag (1=협심증/MI, 2=HTN, 3=DM, 4=뇌졸중, 5=PE)
- `한국 HPV 코호트 자료를 이용한 자_병리검사 (복구됨).CSV` — `병리검사구분`: `조직병리`=재발, `분자병리`/`HPV`=HPV 결과
- `한국 HPV 코호트 자료를 이용한 자_기초임상정보.csv` — height/weight/BP/smoking + `기록일자`

가공 (utf-8-sig, gitignored이 아님): `matched_cohort.csv` → `final_matched_cohort.csv` → `final_matched_outcomes.csv` (Cohort B); Cohort A는 `make_main_figures.py::build_cohort_a_matched`가 in-memory 구성.

## 인코딩·형식 규약

- 원본 CSV: `cp949` (`encoding='cp949'`), 가공 CSV: `utf-8-sig`. `Analysis.R`이 `pathology.csv`로 변환하지만 git 제외.
- 원본 날짜: `YYYYMMDD` 정수 → `pd.to_datetime(..., format='%Y%m%d')`.
- 고위험 HPV 유형 상수: `[16, 18, 31, 33, 45, 52, 58, 35, 39, 51, 56, 59, 66, 68]` (`extract_pathology_outcomes.py::HR_HPV_TYPES`).
- matplotlib 한글: `font.family = ['DejaVu Sans', 'AppleGothic']`, `axes.unicode_minus = False`.

## 스크립트 파이프라인 (`scripts/`)

**전처리 / 매칭** — 순서대로:
1. `build_matched_cohort.py` — Cohort B Step 1 (수술시점/연령/수술방법 1:up-to-5)
2. `build_final_cohort.py` — Cohort B Step 2+3 (index 필터 + fine matching)
3. `extract_pathology_outcomes.py` (또는 `.R`) — 조직병리에서 HSIL/CIN3+ 재발, 분자병리에서 hr-HPV 추출 (`detect_hsil_cin3_recurrence`, `detect_high_risk_hpv`)
4. `extract_outcomes_after_index.py` — Index date 이후 outcome 결합
5. `extract_outcomes_from_diagnosis.py` — 진단 기반 보조 outcome

**일차 분석**:
- `cohort_a_psm.py` / `cohort_a_psm_hr_cif.py` — Cohort A PSM + Fine–Gray + Aalen–Johansen
- `analyze_cohortB_clearance_primary.py` — **Cohort B 공동 일차 결과 (P1 재발 + P2 clearance)**. P2는 pre-vaccine hr-HPV+ subset에 한정, 2-연속음성 정의.
- `rebuild_table2.py` / `rebuild_table3.py` — Cohort A / B HR 표 재생성 (cluster-robust)

**Baseline / Figures**:
- `baseline_table1_unified.py` — Pre/Post matching baseline (A·B 모두, 동일 변수 행)
- `append_table1_clearance_subset.py` — clearance subset baseline append
- `make_main_figures.py` — Figure 1–5 통합 생성 (cohort flow, CIF+forest, KM, vaccine-type, age×FU)
- `make_supfig_S6_sensitivity_forest.py`, `regenerate_love_plots.py`, `make_figure1_pptx.py`

**민감도 분석** — Essential (Sens-A∼E, 본문) vs Appendix (App-1∼10, 부록만). 매핑은 `Analysis_Specifications.md §4.5`:
- `sensitivity_hpv_clearance.py` / `sensitivity_clearance_time_stratified.py` (Sens-A/B)
- `sensitivity_dose_threshold_landmark.py` (Sens-C; immortal-time 보정 landmark)
- `sensitivity_strict_matching.py` (Sens-D)
- `sensitivity_outcome_definition_rigour.py` (Sens-E recurrence DF-interval)
- `sensitivity_exposure_definition.py` — S2 (Rx-code) / S3 (mixed vaccine)
- `sensitivity_hpv_landmark.py`, `sensitivity_hpv_novel_type.py`, `sensitivity_hpv_refined_definition.py`, `sensitivity_vaccine_type_calendar.py`, `sensitivity_analysis*.py`, `sensitivity_age_cutoff.py`, `sensitivity_unadjusted.py`

**보충 파일 / DOCX 빌드**:
- `rebuild_supplementary_clearance.py`, `rebuild_supplementary_misc.py`, `promote_two_negative_primary.py`, `relabel_supplementary*.py`, `slim_supplementary.py`
- `build_docx_artifacts.py` — `docs/Manuscript_Draft.md`를 pandoc으로 manuscript.docx, `Data/*.csv` 표/`Data/*.png` 그림을 묶어 `HPV_tables_figures.docx`, `HPV_supplementary.docx` 생성
- `build_manuscript_docx.py`, `sync_manuscript_docx*.py` — 원고-표/그림 일치 유지

## R 파이프라인 (별도)

`Run.R` → `scripts/extract_pathology_outcomes.R::main()` — Python 버전과 동일 outcome 정의의 R 구현. 원본 → `Data/pathology.csv` 변환은 `Analysis.R` (CP949 → UTF-8).

## 일반 작업 시 유의사항

- **`fine_match_id` 보존**: Cohort B 분석에서 fine-matching 단위가 깨지면 cluster-robust SE와 matched-set 무결성 둘 다 무너진다. 필터링·outcome 결합 시 `fine_match_id` 컬럼을 끝까지 유지.
- **HR 방향 footgun**: P2 (clearance)는 HR > 1이 vaccine-favourable. P1·Cohort A 모든 outcome은 HR < 1이 favourable. 표/그림 라벨링·민감도 비교 시 혼동 주의 (`Analysis_Specifications.md §5` 표).
- **Cohort B 재발 outcome 임계**: 수술 적격은 **HSIL/CIN3+**, post-index 재발 outcome은 **CIN2+** (HSIL+ 또는 invasive). 다른 임계라는 점이 표/원고에 명시되어야 함.
- **Vaccine-type subgroup (n=36 Gardasil 4가)**: 검정력 한계로 본문에서는 forest plot만 사용, "유의 효과"로 해석하지 않음 (예전 CLAUDE.md/legacy 보고서의 `HR=0.395, p=0.004` 결과는 다중비교 미보정·소표본 결과이므로 그대로 인용 금지).
- **Backup docx 처리**: `.gitignore`가 `*.backup.docx`, `*.preslim.docx` 제외. `docs/HPV_manuscript.docx`·`HPV_tables_figures.docx`·`HPV_supplementary.docx`가 정본; 편집 전 backup 생성하는 스크립트들이 있으므로 덮어쓰기 시 git diff 확인.

## 미해결 / 후속

- README §11 항목 (수술 코드 미세 구분, HPV 지속감염 정의 합의, CAD IRB 연장) — 현재 분석은 모두 잠정 정의로 진행.
- App-9 (연령×추적 grid의 30–52y / 2y window 유의 신호) — 표·Limitations로만 보고, 해석 주의.
