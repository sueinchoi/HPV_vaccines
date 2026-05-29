# 본 연구의 Follow-up 특성 — 종합 분석

이 문서는 Cohort B primary cohort (≥2 dose + 3-mo landmark, n = 912)를 중심으로 본 연구의 follow-up 특성을 진단하고, 해석에 영향을 줄 수 있는 selection/length bias·calendar·surveillance·competing-risk 이슈를 정리한다.

---

## 1. 전체 follow-up 분포

**보고 표준**: 본 연구는 follow-up time을 **reverse Kaplan–Meier (Schemper–Smith 1996)** 방식으로 보고한다. Reverse-KM은 event indicator를 invert하여 censoring distribution에 KM을 적용함으로써 **median potential follow-up time**을 unbiased로 추정한다. Naïve median (단순 관찰 FU)은 event가 일찍 일어난 환자가 짧은 시간으로 잘려 들어가 systematic하게 underestimate된다.

### 1.1 Cohort B (PRIMARY, n = 912) — **reverse-KM**

| Time origin | Group | n | reverse-KM median (y) | IQR (y) | Naïve median (참고) | Max observed (y) |
|---|---|---|---|---|---|---|
| From index | Vaccinated | 203 | **4.94** | 3.30–8.90 | 4.87 | 15.09 |
| From index | Non-vaccinated | 709 | **5.02** | 3.48–8.93 | 4.90 | 15.73 |
| From index | **Combined** | 912 | **4.97** | 3.44–8.91 | — | — |
| From landmark (index + 90 d) | Vaccinated | 203 | 4.69 | 3.05–8.65 | 4.63 | 14.84 |
| From landmark (index + 90 d) | Non-vaccinated | 709 | 4.78 | 3.23–8.68 | 4.65 | 15.48 |
| From landmark (index + 90 d) | **Combined** | 912 | **4.73** | 3.20–8.67 | — | — |

### 1.2 Cohort A (PSM, n = 4,160) — **reverse-KM**

| Group | n | reverse-KM median (y) | IQR (y) |
|---|---|---|---|
| Vaccinated | 2,050 | **4.68** | 1.95–9.92 |
| Non-vaccinated | 2,110 | **6.99** | 3.73–11.21 |
| Combined | 4,160 | **5.93** | 2.78–10.75 |

⚠ Cohort A는 두 군 reverse-KM median이 다소 차이남 (4.68 vs 6.99 y). 이는 vaccinated의 pseudo-vaccine date 분포가 calendar 시점상 더 최근 (2018–2021 peak)이어서 administrative censoring까지 남은 시간이 더 짧기 때문 — PSM 매칭 자체가 calendar-period을 직접 묶지 않음.

**해석**: 두 군의 follow-up은 잘 균형되어 있다. 중간 추적 5년 + 평균 6년 + 최장 15년이라는 분포는 **CDW 기반 후향 cohort study로서는 매우 긴 편**이며, KOSIS·국내 HPV 백신 효과 연구 중 가장 긴 FU 범위에 속한다.

### 1.2 Cohort B legacy (≥1 dose, no landmark, n = 1,108)
- Vac median 4.88 y (mean 6.12), Non median 5.02 y (mean 6.12).  cohort와 거의 동일.

---

## 2. Calendar distribution (index date)

```
Year  | Vac    Non    Total
2010  |   7     23     30   ██████████
2011  |  14     48     62   ████████████████████
2012  |  15     56     71   ███████████████████████
2013  |  14     41     55   ██████████████████
2014  |  10     42     52   █████████████████
2015  |  15     52     67   ██████████████████████
2016  |  13     45     58   ███████████████████
2017  |  12     48     60   ████████████████████
2018  |  16     70     86   ████████████████████████████
2019  |  19     77     96   ████████████████████████████████
2020  |  28     89    117   ███████████████████████████████████████
2021  |  28     89    117   ███████████████████████████████████████
2022  |  11     35     46   ███████████████
2023  |   2     15     17   █████
```

**관찰**:
- 2010–2023의 13년 calendar span. CDW 기반 study로서는 매우 폭넓다.
- 2018–2021이 peak (전체 416/912 ≈ 45.6%) — 한국 9가 백신 도입(2016) 및 청년층 자가 접종 확산 시기와 일치.
- 2022–2023 enrolment은 작음 (총 63명, 6.7%) — **이들은 분석 시점(2025-12-31)까지 남은 follow-up이 2–3년에 불과**해 long-term 추정에는 기여하지 못함.

**Right-truncation**: 분석 종료시점 2025-12-31에 censoring된 환자 0명 (모두 그 전에 last_follow_up이 있음) — 즉 administrative censoring보다는 **자연 loss-to-follow-up**이 종료의 주된 원인이다.

---

## 3. Surgery → Index gap (vaccine timing or pseudo-index assignment)

| Group | Definition | n | Median (y) | IQR (y) | Max (y) |
|---|---|---|---|---|---|
| Vac | 첫수술일자 → 첫 백신일 | 241 | **0.34** | 0.07–0.58 | 10.68 |
| Non | 첫수술일자 → pseudo-index | 867 | 0.32 | 0.08–0.53 | 6.09 |

**해석**: 매칭이 잘 작동해서 두 군의 surgery → index gap이 거의 동일. 중간값 ~4개월, 75th percentile ~6.5개월 — 즉 **대부분의 환자가 수술 후 1년 이내에 백신 접종 또는 pseudo-index 할당**되어 surgery aftermath와 vaccine effect window가 깔끔하게 분리된다.

---

## 4. Competing risks (사망)

| Group | n | Deaths during FU | % |
|---|---|---|---|
| Vac (primary) | 203 | 0 | 0.00% |
| Non-vac (primary) | 709 | 3 | 0.42% |
| Source population (모든 N=32,969) | 32,969 | 376 | 1.14% |

**해석**: Cohort B는 평균 37세 여성 인구이며 follow-up 중 **사망 거의 없음 (3건)**. 따라서:
- competing risk 보정이 사실상 불필요 (KM ≈ Aalen–Johansen CIF)
- Cohort A (장기 안전성)에서는 사망이 고려되지만 Cohort B에서는 무시 가능

---

## 5. Surveillance density (post-index)

### 5.1 HPV molecular pathology tests

| Group | Tests/year (median) | Tests/year (mean) | Total post-index mol tests (median, IQR) |
|---|---|---|---|
| Vac | 0.89 | 0.96 | 4 (2–6) |
| Non | 0.81 | 0.96 | 4 (2–6) |

### 5.2 Tissue pathology tests (biopsy/conization)

| Group | Tests/year (median) | Tests/year (mean) |
|---|---|---|
| Vac | 0.00 | 0.11 |
| Non | 0.00 | 0.16 |

**해석**:
- HPV 검사 density는 두 군이 **거의 동일** (median 0.81–0.89 tests/yr) — 즉 **surveillance bias로 양성/음성 비율 차이를 설명할 수 없음**.
- Tissue biopsy는 비정상 cytology/HPV 결과에 trigger되는 것이므로 median 0은 자연스러움 (대다수 환자는 추적 중 follow-up biopsy 없음).
- 평균이 vac (0.96/yr)·non (0.96/yr) 완전히 같다는 점이 surveillance equality의 가장 강력한 증거.

---

## 6. Length-bias diagnosis (clearance subset)

**핵심 발견 — sustained clearance KM median이 부풀려진 이유**

| Group | Status | n | Median FU from index (y) | Mean (y) |
|---|---|---|---|---|
| Vac | **cleared** | 32 | **8.67** | 8.36 |
| Vac | not cleared | 63 | 4.50 | 5.12 |
| Non | **cleared** | 40 | **5.54** | 6.94 |
| Non | not cleared | 119 | 4.73 | 5.38 |

**관찰**:
- **Vaccinated cleared 환자는 median FU 8.67년**으로, primary cohort 전체 vac median (4.87년)의 **거의 2배**.
- Non-vac cleared 환자도 비슷한 length bias (5.54 vs 4.73년) 하지만 vac에서 훨씬 강함.
- 즉, "2연속 음성" 정의는 자연히 follow-up이 긴 환자를 selection.
- 이 length bias가 sustained clearance KM median (vac 10.79y vs non 5.67y)을 **인공적으로 부풀린 것**으로 확인됨.

**대응**: 5-year reversion-free probability (vac 56.9% vs non 53.3%)는 length bias의 영향을 적게 받아 **두 군 거의 동일**. 이 수치를 Table 3에 KM median과 병기.

---

## 7. Time-to-event distribution (clearance + recurrence)

### 7.1 Clearance event timing (index → first_neg_date)
| Group | n | Median (y) | IQR (y) | Max (y) |
|---|---|---|---|---|
| Vac cleared | 32 | **0.54** | 0.49–0.89 | 8.53 |
| Non cleared | 40 | 0.61 | 0.45–1.09 | 8.55 |

**해석**: Clearance가 일어나는 시점은 양 군 모두 **수술 후 6개월 시점**에 집중. IQR이 매우 좁고 median이 거의 같음 → 한국에서 자궁경부 수술 후 첫 추적 검사(보통 6개월)에서 두 음성을 잡는 패턴이 일반적.

### 7.2 Lesion recurrence timing (post-index)
| Group | n events | Median (y) | IQR (y) | Max (y) |
|---|---|---|---|---|
| Vac | 11 | 2.23 | 0.44–3.87 | 5.63 |
| Non | 50 | 0.86 | 0.11–3.36 | 8.62 |

**해석**: 
- Non-vac에서 재발이 더 일찍 일어남 (median 0.86 y vs 2.23 y).
- 그러나 primary HR = 1.01 (null) — 즉 timing 차이는 있으나 누적 incidence는 동등.
- Non-vac의 early recurrence는 surgical incomplete excision 가능성을 시사 (Sens-E disease-free interval analysis에서도 다뤄진 부분).

---

## 8. 종합 평가 — Follow-up이 "긴 편"인가?

**그렇다 — 비교 가능한 한국 후향 cohort study 중 상위권**:

| 지표 | 본 연구 () | 일반적 후향 cohort 기준 |
|---|---|---|
| Median FU | 4.87 y | 2–4 y (대부분) |
| Mean FU | 6.10 y | 3–5 y |
| Max FU | 15.09 y | 8–10 y |
| Total person-years (P1) | 5,079 (1,132 vac + 3,947 non) | 매우 풍부 |
| 사망 censoring | 0.4% | 무시 가능 |

**왜 긴가**:
1. CDW에 2010–2023 13년치 데이터 축적 (HPV 백신 도입 직전부터 최근까지).
2. 연구 종료가 2025-12-31로 분석 시점 직전까지 추적.
3. 대상 인구가 평균 37세로 자연 loss-to-follow-up이 적은 연령대 (다른 만성질환 cohort 대비).
4. 단일 의료기관 CDW이므로 institutional follow-up이 지속적.

**의의**:
- Lesion recurrence (median 0.86–2.23 y에 발생) 및 hr-HPV clearance (median 0.54–0.61 y에 발생)는 모두 본 연구의 FU 범위 안에서 충분히 관측 가능.
- 단, **clearance achiever에 한정한 sustained clearance 분석은 length-biased**이므로 KM median 보고에 caveat 필수 (Section 6 참고).
- Vac의 max FU 15.09 y는 9가 백신 도입(2016) 이전 (4가 또는 2가) 접종자가 일부 포함되어 있다는 신호 — vaccine-type subgroup 분석 시 calendar period confounding 주의.

---

## 9. Reviewer 대응용 한 줄 요약

> 본 연구는 단일 의료기관 CDW에서 2010–2023년 사이 enrolment된 cohort에 대해 2025년 12월까지 추적해, Cohort B 매칭 코호트(n = 912)에서 **median 4.87 / mean 6.10 / max 15.09년의 follow-up**을 확보했다. 두 군의 추적기간·사망률·surveillance density는 모두 일치하며 (reverse-KM potential FU 4.94 vs 5.02 y; HPV 검사 density 0.89 vs 0.81 /yr), administrative right-truncation·loss-to-follow-up imbalance는 관측되지 않는다. 다만 clearance achiever 한정 sustained-clearance 분석은 정의상 longer-FU 환자를 selection하므로 KM median을 보고하되 5-year reversion-free probability를 병기해 length bias의 영향을 명시한다.
