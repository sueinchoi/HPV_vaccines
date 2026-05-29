# Figure 1 — Cohort Selection Flow Diagram

**File:** `Data/Figure1_CohortSelection.png`

## Suggested figure legend (manuscript-ready)

> **Figure 1. Construction of the two analytic cohorts from a single source population (Korean HPV cohort, 2009–2024).**
> A total of 32,969 women were enrolled prospectively in the Korean HPV cohort between 2009 and 2024. Receipt of HPV vaccination was ascertained from the Clinical Data Warehouse (CDW) prescription file by joint matching on Korean and English drug names (가다실 / 서바릭스 / Gardasil / Cervarix / HPV vaccine) and on the institution-specific drug-code prefixes (DV-9HPF, DV-HPF, DV-JHP); the two ascertainment methods identified an identical set of 2,156 vaccinated women (concordance 100%). The unvaccinated comparator pool comprised the remaining 30,813 women. Two analytic cohorts were derived independently to address complementary research questions. **Cohort A** (long-term safety) included the entire source population and was constructed by 1:1 propensity-score matching with a caliper of 0.2 × SD(logit propensity score) without replacement, followed by the pre-specified primary exposure filter (≥2 documented vaccine prescription dates and a symmetric 3-month landmark, with the matched pair preserved as the unit of integrity), yielding a final analytic Cohort A of **2,776 women (1,396 vaccinated and 1,380 unvaccinated)**. **Cohort B** (post-surgical efficacy) was restricted to the 6,890 women who underwent cervical conization or hysterectomy for histologically confirmed HSIL/CIN3 or higher disease, and was constructed by a two-step variable-ratio matching procedure: an initial 1:up-to-5 match on surgical procedure (exact), surgical year (±1 year), and age at surgery (±5 years), an index-date eligibility filter (index date ≤ 31 December 2020 and ≥2 follow-up records), and a fine 1:up-to-4 match on age at index (±5 years), body-mass index (±3 kg/m²), and surgical year (±1 year), followed by the same primary exposure filter with the fine-matched set preserved as the unit of integrity, yielding a final analytic Cohort B of **912 women (203 vaccinated and 709 unvaccinated)**.
>
> All matching used a greedy nearest-neighbour algorithm without replacement (random seed = 42); the requested ratios (5 and 4) represent the maximum number of controls per vaccinated case, with cases retaining fewer controls when the eligible pool was depleted or when no comparable controls existed within the specified tolerances.
>
> Index date for vaccinated women was the date of first HPV vaccine prescription. For unvaccinated women, a pseudo index date was assigned: in Cohort A by random sampling from the empirical distribution of vaccine dates (preserving the temporal frame of follow-up), and in Cohort B as the unvaccinated woman's surgery date plus the surgery-to-vaccination interval of her matched vaccinated partner (preserving the immortal-time correspondence between groups). PSM = propensity-score matching.

## Methodological footnotes (for Methods, Limitations, or Supplement)

The following points are *not* required in the figure caption itself but should appear once in the manuscript so that a reader can fully reconstruct the cohort.

### a. Vaccine-exposure ascertainment (validation)
- 2,156 women received ≥1 HPV vaccine prescription (5,514 dose-level rows total).
- Drug-code distribution: `DV-JHP0.5F` (Cervarix, 2-valent) 2,323 doses; `DV-9HPF0.5` (Gardasil 9, 9-valent) 2,265 doses; `DV-HPF0.5` (Gardasil, 4-valent) 885 doses; minor variants (`-FR` free vaccination, `-FJ` employee/family, `-HPJ` legacy) account for 41 doses.
- Per-patient dose distribution: median 3 doses (IQR 2–3, range 1–6); 80% completed the standard three-dose schedule.
- The name-string and drug-code mask agreed on the same 2,156 patients (no name-only or code-only exposures), giving high confidence in exposure ascertainment.

### b. Mixed vaccine-type recipients
- 33 of 2,156 vaccinated women (1.5%) received doses of more than one vaccine product (Gardasil 9 + Gardasil 4-valent: n = 21; Gardasil 9 + Cervarix: n = 10; Gardasil 4-valent + Cervarix: n = 2). For all subgroup analyses by vaccine type, the product was attributed by the first-administered dose; reclassification by last dose changed the assignment for these 33 women only and did not alter inferences.

### c. Cohort B dropout pathway from 2,156 vaccinated to 203 final
- The reduction is by design: Cohort B is a post-surgical efficacy cohort, so women without cervical conization or hysterectomy (1,590 / 2,156, 74%) are categorically ineligible. Of the 566 vaccinated women who had also undergone cervical surgery, 155 could not be matched at the initial 1:up-to-5 step (no eligible unvaccinated comparator within the surgery-method × ±1-year × ±5-year tolerance windows), and a further 170 (30%) could not be matched at the fine 1:up-to-4 step (no comparator within the index-age × BMI × surgery-year tolerances or the eligible pool was exhausted), leaving 241 vaccinated cases (with 867 unvaccinated controls) entering the primary exposure filter. Application of the ≥2-dose-plus-3-month-landmark filter with matched-set integrity then removed 38 further vaccinated cases (and their attached controls) — 36 because the vaccinated case received only one documented dose, and an additional small number because the vaccinated case experienced a recurrence or hr-HPV event before the 3-month landmark — yielding the final primary analytic Cohort B of **203 vaccinated and 709 unvaccinated women (n = 912)**.

### d. Choice of variable-ratio over fixed-ratio matching
- A strict fixed 1:4 fine-matching alternative was prespecified as a sensitivity analysis (Sup Table on strict matching). Strict matching dropped a non-trivial share of vaccinated cases while yielding essentially identical lesion-recurrence hazard-ratio estimates. Variable-ratio matching was preferred for the primary analysis because (i) it preserves all vaccinated cases for which any acceptable comparator exists — important given the limited number of vaccinated surgical patients and the small absolute event count, and (ii) cluster-robust standard errors based on the matched-set identifier appropriately accommodate the unequal cluster sizes.

### e. Power and statistical caveats
- The detectable hazard ratio at 80% power and α = 0.05 is approximately 1.95 for lesion recurrence (n = 912; 43 events) and approximately 1.70 for hr-HPV clearance on the n = 233 pre-vaccine hr-HPV-positive subset (59 two-consecutive-negative clearance events), indicating that the primary analysis is well powered only to detect large effects on either co-primary outcome.

## Suggested abbreviation key (for figure or footnote)
- HPV = human papillomavirus
- CDW = clinical data warehouse
- PSM = propensity-score matching
- HSIL/CIN2+ = high-grade squamous intraepithelial lesion / cervical intraepithelial neoplasia grade 2 or higher (post-index recurrence outcome). Surgical eligibility for entry was confirmed HSIL/CIN3+, the standard threshold for cervical excision.
- BMI = body-mass index
- SBP / DBP = systolic / diastolic blood pressure
- SD = standard deviation
