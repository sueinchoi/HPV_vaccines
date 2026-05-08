# CLAUDE.md

이 저장소는 한국 HPV 코호트 데이터를 이용해 자궁경부 수술 후 **HPV 백신 접종의 병변 재발 및 HPV 재감염 예방 효과**를 평가한 후향적 매칭 코호트 연구의 분석 코드와 결과물 저장소이다. (목표: 2026 부인종양학회 포스터 발표)

## 프로젝트 개요

- **연구 설계**: Retrospective matched cohort study
- **연구 대상**: 자궁경부 상피내 병변(ASCUS/LSIL → HSIL/CIN3 또는 자궁경부암 진행) 수술(원추절제술/자궁절제술) 환자
- **노출**: 수술 후 HPV 예방접종 여부
- **Index date**: 접종군은 백신 접종일, 비접종군은 매칭된 접종군의 "수술-접종 간격(T)"을 적용한 pseudo index date
- **추적 종료**: outcome 발생일 / 사망 / 자격상실 / 2025-12-31 중 가장 빠른 날짜

### 결과변수
- **Primary**: ① 병변 재발(HSIL/CIN3 이상 조직검사 확인), ② 새로운 고위험 HPV 감염 (유형 16, 18, 31, 33, 45, 52, 58, 35, 39, 51, 56, 59, 66, 68)
- **Secondary**: 당뇨(HbA1c ≥ 6.5%), 고혈압(SBP≥140 또는 DBP≥90, 2회 이상)

### 매칭 전략
1. **1차 매칭 (1:5)**: 수술시점(±1년), 나이(±5세), 수술방법(exact) → 접종군 411 / 비접종군 1,815
2. **Index date 필터링**: 접종군 411 / 비접종군 1,797 (18명 제외)
3. **2차 Fine Matching (1:4)**: Index 시점 나이, BMI, 수술연도 → **최종 접종군 241 / 비접종군 867 (총 1,108명)**

## 디렉토리 구조

```
HPV_vaccines/
├── README.md                  # 연구 계획서 (목적, 설계, 매칭, 분석 방법)
├── Run.R                      # R 파이프라인 진입점 (pathology outcomes)
├── Analysis.R                 # 원본 CP949 CSV → UTF-8 변환 + 샘플 추출
├── Data/                      # 원본 CSV (CP949), 가공 CSV, 결과 figure/표
├── scripts/                   # Python/R 분석 스크립트
└── .gitignore                 # 원본 병리 CSV는 git 제외
```

### 핵심 데이터 파일 (`Data/`)
- 원본 (CP949): `한국 HPV 코호트 자료를 이용한 자_*.csv` (코호트, 진단정보, 수술처방, 처방정보, 진단검사Lab, 병리검사, 기초임상정보)
- `한국 HPV 코호트 자료를 이용한 자_수술처방_수술종류구분완료.csv` — 원추절제술/자궁절제술/제외 분류 완료된 버전
- 가공: `matched_cohort.csv` (1차 매칭) → `final_matched_cohort.csv` (Fine matching) → `final_matched_outcomes.csv` (결과 변수 결합)
- 보고서: `HPV_vaccine_study_report_vaccine.docx` ⭐ **최종 보고서 (백신 종류별 분석 포함)**, `HPV_vaccine_study_report.docx` (이전 버전)

### 스크립트 실행 순서 (`scripts/`)
1. `build_matched_cohort.py` — 1차 매칭 코호트 구축 (수술시점/나이/수술방법)
2. `build_final_cohort.py` — Index date 필터링 + Fine Matching (나이/BMI/수술연도)
3. `extract_pathology_outcomes.py` (또는 `.R`) — 병리 데이터에서 HSIL/CIN3+ 재발, 고위험 HPV 추출
4. `extract_outcomes_after_index.py` — Index date 이후 outcome 추출
5. `extract_outcomes_from_diagnosis.py` — 진단 데이터 기반 보조 outcome
6. `analyze_cohort.py` — SMD/Love plot, Cox PH 메인 분석
7. `vaccine_type_analysis.py` — Gardasil9/Cervarix/Gardasil 백신 종류별 분석
8. 민감도 분석: `sensitivity_analysis.py`, `sensitivity_analysis_v2.py`, `sensitivity_analysis_both.py`, `sensitivity_age_cutoff.py`, `sensitivity_unadjusted.py`

모든 Python 스크립트에 `RANDOM_SEED = 42` 고정. 매칭/분석은 한글 컬럼명을 그대로 사용하며 인코딩 일관성에 주의 (`utf-8-sig` for 가공, `cp949` for 원본).

## 최종 결과 요약 (`HPV_vaccine_study_report_vaccine.docx`)

### 1. 코호트
- 전체 수술 환자 6,890명 → 1차 매칭 후 2,226명 → 최종 **1,108명** (접종 241 / 비접종 867)
- 평균 Index 나이 37.2세, 평균 BMI 22.3, 평균 수술연도 2016, 평균 추적 ~2,236일

### 2. 백신 종류 분포 (접종군 241명)
| 백신 | N (%) | 평균 수술-접종 간격 |
|---|---|---|
| Gardasil9 (9가) | 128 (53.1%) | 265.1일 (8.8개월) |
| Cervarix (2가) | 77 (32.0%) | 148.4일 (4.9개월) |
| Gardasil (4가) | 36 (14.9%) | 167.7일 (5.6개월) |
| **전체** | **241** | **213.2일 (7.1개월)** |

### 3. 메인 결과 (Cox PH, 연령 보정)
| Outcome | 접종군 | 비접종군 | HR (95% CI) | p |
|---|---|---|---|---|
| 병변 재발 | 13/241 (5.4%) | 57/867 (6.6%) | 0.795 (0.435–1.454) | 0.4566 |
| HPV 재감염 | 149/241 (61.8%) | 546/867 (63.0%) | 0.908 (0.757–1.088) | 0.2963 |

→ 전체 코호트에서는 **유의한 보호 효과 없음**

### 4. 유의한 Subgroup
- **연령별 (병변 재발, 2년 추적)**: 30–52세 HR=0.230 (0.055–0.963), p=0.044 / 20–52세 HR=0.299, p=0.045 / 30–50세 HR=0.238, p=0.050
- **백신 종류별 (HPV 재감염)**: **Gardasil(4가) HR=0.395 (0.209–0.745), p=0.004** ← 유일한 유의 결과
  - Gardasil9: HR=0.909, p=0.434 (NS)
  - Cervarix: HR=1.276, p=0.126 (NS)

### 5. 결론 및 제한점
- 전체 코호트에서는 통계적으로 유의한 보호 효과 미관찰
- 30–52세 2년 추적에서 병변 재발 유의 감소
- **Gardasil(4가)에서 HPV 재감염 유의 보호 효과**, Gardasil9/Cervarix는 NS
- 한계: 후향적 설계, 적은 사건 수로 검정력 한계, 다중 비교 문제, Gardasil 표본 작음(n=36), 백신 접종 동기 정보 부재

## 작업 시 유의사항

- **인코딩**: 원본 CSV는 `cp949` (encoding='cp949'), 가공 CSV는 `utf-8-sig`. `Analysis.R`이 변환을 담당하지만 `pathology.csv`는 git 제외(`.gitignore`)
- **날짜 컬럼**: 원본은 `YYYYMMDD` 정수형 (`format='%Y%m%d'`로 파싱), 가공 후 datetime
- **고위험 HPV 유형**: `[16, 18, 31, 33, 45, 52, 58, 35, 39, 51, 56, 59, 66, 68]` (스크립트 상수)
- **재현성**: 모든 매칭/sampling은 `np.random.seed(42)`
- **수술 분류**: `_수술처방_수술종류구분완료.csv` 사용 (수동 분류 완료본). "제외" 분류는 코호트에서 빠짐
- **한글 폰트**: `matplotlib`에 `AppleGothic`/`NanumGothic` fallback 설정됨

## 미해결 사항 (README §11)
- 수술 코드 구분(원추/자궁절제술) 추가 검토 — 현재는 _수술종류구분완료.csv로 갈음
- HPV 지속감염 정의 (협의 필요)
- CAD 관련 진단 추가 (IRB 연장 심의 후)
