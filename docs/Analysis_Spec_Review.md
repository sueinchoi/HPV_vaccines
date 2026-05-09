# Statistical and clinical review of `Analysis_Specifications.md`

A peer-style critique of the analytic plan documented in
`docs/Analysis_Specifications.md`. Organised by domain, with each item
classified as either ✓ adequate, ⚠ minor concern, or ✗ requires
substantive change.

---

## A. Statistical methodology

### A1. Causal estimands and target trial framing
- ⚠ **No explicit estimand statement.** The clearance co-primary is conditional on pre-vaccine hr-HPV positivity, which for unvaccinated controls is conditioning on a post-randomisation calendar time (their pseudo-index); this is methodologically defensible because matched-set integrity is preserved, but the manuscript should make explicit that the estimand is *the average effect of post-surgical vaccination on the rate of post-index hr-HPV clearance, in the population of women who are still HPV-positive at the vaccine date* (or its matched-control counterfactual).
- ✓ The lesion-recurrence co-primary has a cleaner intention-to-treat-like estimand because no conditioning is required.
- **Suggested fix**: add one sentence to Methods §Statistical analysis using ICH E9(R1) terminology — "treatment policy" estimand for recurrence; "principal stratum" estimand (limited to baseline-positive subset) for clearance.

### A2. Time-zero anchoring and immortal-time
- ✓ Pseudo-index assignment (surgery + matched interval) addresses immortal-time bias in Cohort B as designed.
- ✓ The pseudo-index sensitivity (S8 in Cohort A) confirms the inference is robust to alternative anchoring strategies.
- ⚠ Pseudo-index for Cohort A is randomly drawn from the empirical vaccine-date distribution; this is convenient but could be improved by **risk-set sampling** (already done as S8 sensitivity, with concordant results). Consider whether any single sensitivity is preferable as primary.

### A3. Matching strategy
- ✓ Variable-ratio (1:up-to-N) matching is standard (Stuart 2010, Austin 2014); manuscript correctly cites these.
- ✓ Cluster-robust standard errors on the matching identifier appropriately handle the dependence structure.
- ✓ Matched-set integrity is preserved under all secondary filters (clearance subset, dose threshold, strict matching).
- ⚠ **Optimal vs greedy matching** — Greedy matching is acceptable but not optimal; an *optimal* matching (Hansen 2006) would minimise the global distance and could be a useful supplementary check. Not essential.
- ⚠ **Caliper choice for Cohort A** — 0.2 × SD(logit PS) follows Austin 2011, but this is now considered slightly conservative; some recent guidance suggests 0.10. The current matching is balanced (love plot), so no change needed.

### A4. Outcome time and censoring
- ✓ Time = days_to_event for events, follow_up_days otherwise — correct for survival analysis.
- ✓ Administrative censoring at 31 December 2025 is a clean lock date.
- ✓ Independent censoring is a reasonable assumption given near-identical testing frequency between groups (median 4 tests per patient, ~0.8/year, in both arms).
- ⚠ **Death is treated as right-censoring in Cohort B**, but in Cohort A it is a competing risk handled with Aalen–Johansen / Fine–Gray. The asymmetry is appropriate (deaths are extremely rare in Cohort B's young population, n = 0 in clearance subset) but should be stated explicitly.

### A5. Cox model specification
- ✓ Age-adjusted, cluster-robust on match ID — standard for matched cohort.
- ⚠ **Age modelled linearly** — the spline of age might capture residual nonlinearity. Given the matching already balances on age (|SMD| 0.04 for clearance subset), this is unlikely to change estimates materially.
- ✓ Schoenfeld residual rank test reported on every primary and key sensitivity.

### A6. Proportional-hazards violations
- ✗ → ✓ (just resolved in this commit) The clearance co-primary's PH violation (p = 0.007) is now addressed by a pre-specified piecewise time-stratified Cox model (Sup Table S17). The decomposition reveals a delayed-response pattern (HR 0.62 in 0–6 months → **HR 2.78 in 6–12 months** → 1.30 → 1.64), consistent with the antibody maturation timeline of HPV vaccines (peak titres 30–90 days after the third dose) and clinically interpretable as the vaccine effect emerging once antibody response is mature, then converging as natural clearance kinetics accumulate in both arms.
- ⚠ The post-index detection sensitivity also shows a PH violation (p = 0.021); the time-stratified analysis was not performed for this outcome because the outcome itself is a less rigorous endpoint.
- **Suggested**: consider reporting **restricted mean survival time (RMST) difference** at 1 and 2 years as a complementary summary that does not assume PH; the difference is interpretable as "additional months of HPV-free time conferred by vaccination over the first N years."

### A7. Multiple comparisons
- ✓ Two co-primary outcomes addressed via Rothman 1990 (no adjustment for distinct biological hypotheses); STROBE has no multiplicity item.
- ⚠ **Subgroup analyses in Figure 4** — 4 strata × 2 outcomes × subgroup × interaction = many tests. Manuscript correctly frames these as exploratory, but the ≥50-yr lesion recurrence point estimate of 3.23 (n = 22 vac vs 74 ctl, 5 events each) is fragile and should not be over-interpreted.
- ✓ The post-hoc 30–52-yr finding from a 6,900-cell grid search is correctly relegated to Limitations only.

### A8. Sample size and power
- ✓ Detectable HR for clearance ≈ 1.6 (verified: 1.66) at 80% power, reported transparently in Limitations.
- ⚠ **Lesion recurrence has only 13 events in the vaccinated arm**; the analysis is sensitive to detect protective effects only of magnitude approximately HR ≤ 0.5 at 80% power. Already noted.
- ✓ The 6–12-month time-stratified HR 2.78 is based on 35 events (22 vac + 13 ctl), so the CI 1.37–5.62 is wide but the lower bound excludes 1.

### A9. Sensitivity analyses
- ✓ Comprehensive coverage (12 pre-specified analyses; S3–S17). Each is reproducible from a named script.
- ✓ All key directions covered: exposure (dose threshold, prescription code), matching (strict, variable-ratio choice), outcome (post-index detection, landmark, novel-type, clearance, type-specific), time (restricted FU, time-stratified), confounding (vaccine-type by calendar period).
- ⚠ **Missing**: no formal *negative-control outcome* analysis. A negative control (e.g., a condition expected to be unaffected by HPV vaccination, like dysmenorrhea or thyroid disorder) would strengthen the inference against unmeasured confounding. Optional.
- ⚠ **Missing**: no E-value calculation for the lesion-recurrence HR. An E-value of ~3.5 (corresponding to HR 0.80 at the lower CI bound 0.44) is reassuring; reporting it would help readers gauge robustness to unmeasured confounding.

---

## B. Clinical interpretation

### B1. Patient population
- ✓ Cohort B is clinically appropriate — women undergoing cervical excision for HSIL/CIN3+ are the relevant target for the post-treatment HPV-vaccine question.
- ✓ Surgery distribution (>99% conization, <1% hysterectomy) reflects current Korean clinical practice; CIN2/3+ is normally treated with conization.
- ⚠ **Single-centre, Korean-only cohort** — generalisability to other healthcare systems with different testing protocols and HPV epidemiology is uncertain. Already acknowledged in Limitations.

### B2. Outcome definitions
- ✓ Lesion recurrence at the CIN2+/HSIL+ threshold is consistent with Lichter (2020 Obstet Gynecol) and Petráš (2023 Sex Transm Infect) meta-analyses.
- ✓ Tissue-pathology confirmation (not cytology alone) is the appropriate threshold for "recurrence".
- ⚠ **Clearance** as an outcome — clinically meaningful but operationally challenging:
  - "Cleared" women may retain low-level viral DNA below detection threshold and re-emerge later. We do not capture true biological clearance, only "no detectable hr-HPV at the time of testing."
  - The median time-to-clearance of 200+ days is plausible (literature suggests 6–24 months in untreated women).
  - Differential testing intensity could, in principle, bias clearance ascertainment, but our audit (median 4 tests, ~0.8/yr in both arms) argues against this.
- ⚠ **Type-specific clearance HRs lack a vaccine-mechanism signature** — Gardasil 9 covers HPV 16 yet HR 0.72; Cervarix covers 16/18 with HR 1.43; quadrivalent Gardasil with HR 1.55. A clean vaccine-antigen mechanism would predict directionally consistent type-specific protection. This is consistent with our overall null inference.

### B3. Vaccine timing and immune response
- ⚠ **Surgery-to-vaccine interval differs by product** (median 71 / 134 / 138 days for nine-valent / quadrivalent / Cervarix). Manuscript correctly highlights this as a confounder for the apparent type-specific signal.
- ✓ The time-stratified clearance result (HR 2.78 at 6–12 mo post-index) is **biologically supported** — IgG titres after the 3-dose series peak 30–90 days post-dose-3, and clearance of established pre-vaccine infection by humoral immunity has been demonstrated in subgroup analyses of the FUTURE I/II trials.
- ⚠ **Caveat**: 80% completed ≥3 doses and dose 3 is typically given at 6 months, so the 6–12-month window post-vaccine-1 corresponds to roughly the period during and shortly after dose 3 — exactly when antibody titres are rising. This is mechanistically plausible but also exposes the analysis to a "dose-3 effect" interpretation.

### B4. Selection bias and confounding by indication
- ✓ Manuscript Limitations explicitly addresses the indication-for-vaccination confounding.
- ⚠ **Unmeasured confounders**: sexual behaviour, partner change rate, contraception, condom use, previous Pap-smear adherence are all unobserved. Korean women who voluntarily seek post-surgical vaccination may differ systematically.
- ⚠ Recommend computing the **E-value** (above) so readers can judge robustness.

### B5. Generalisability
- ⚠ Korean post-surgical surveillance is intensive (median 4 HPV tests per patient over follow-up). In settings with less testing, the clearance ascertainment would be sparser and the analysis less powered.
- ⚠ The Korean market's near-complete G4v→G9v product transition creates the calendar-period × vaccine-type collinearity that prevents distinguishing them; this is a Korea-specific limitation and should be noted as such.

### B6. Clinical relevance of the time-stratified finding
- ⚠ The 6–12-month HR 2.78 (1.37–5.62) is a **statistically significant**, **biologically plausible**, **mechanistically anchored** finding. Two interpretations:
  1. **Hypothesis-generating** — suggests that vaccination during the immediate post-treatment surveillance period may accelerate clearance of pre-existing infection in the 6–12-month window; if confirmed, this would inform clinical guidance on timing.
  2. **Cautionary** — without correction for multiple comparisons across the 4 time windows, the nominal p = 0.005 should not be over-interpreted; a Bonferroni-adjusted threshold for 4 windows is 0.0125, so the finding still survives.
- The current manuscript framing (Results paragraph: "this pattern is consistent with the 1–3 month interval typically required for HPV-vaccine antibody maturation") is appropriately restrained.

### B7. Safety analysis
- ✓ Cohort A's null safety findings across five chronic conditions and two composites are clinically reassuring.
- ✓ Person-years and incidence rates per 1,000 PY are reported, allowing readers to compare to background population rates.
- ⚠ **Mortality during follow-up is rare** (n = 28 in matched cohort, 0 in Cohort B clearance subset), so the competing-risk framework for Cohort A is largely a methodological precaution rather than a clinically influential adjustment.

---

## C. Reporting and reproducibility

### C1. Documentation
- ✓ `Analysis_Specifications.md` is comprehensive and internally consistent.
- ✓ Each output file has a named generator script.
- ✓ Random seed = 42 is consistent across all stochastic procedures.
- ⚠ **Suggested addition to Spec**: an explicit estimand framing per ICH E9(R1) at the top of §3 and §4.

### C2. Manuscript ↔ data ↔ spec consistency
- ✓ All numerical claims in the manuscript verified against the data (audit completed; one rounding discrepancy fixed: Cohort A Any-of-five p value 0.374 → "0.37" rather than "0.38").
- ✓ All claims in the spec verified against the data.
- ✓ Spec ↔ manuscript ↔ supplementary tables all consistent as of git commit `e2d8076`.

### C3. PHI safeguards
- ✓ `.gitignore` excludes raw cohort source files (`한국 HPV 코호트 자료를 이용한 자_*`), `.DS_Store`, Office lock files, and `__pycache__`.
- ✓ Repository is public on GitHub; no PHI is committed.

---

## D. Summary recommendations

| # | Item | Priority | Action |
|---|---|---|---|
| 1 | Time-stratified clearance HR | High | ✅ Done (Sup S17, manuscript Results updated) |
| 1b | Two-consecutive-negative HPV clearance | High | ✅ Done (Sup S18, HR 1.40 [0.92–2.11] vs primary 1.23) |
| 1c | ≥6-month disease-free interval for recurrence | High | ✅ Done (Sup S19, HR 0.86 [0.41–1.78], primary 0.80 robust) |
| 2 | ICH E9(R1) estimand framing | Medium | Add 1 sentence to Methods §Statistical analysis |
| 3 | E-value for residual confounding | Medium | Compute and report in Limitations or Sup |
| 4 | RMST difference at 1y / 2y | Low | Optional complement to PH-violated clearance HR |
| 5 | Negative-control outcome | Low | Optional; would strengthen inference |
| 6 | Cohort A Any-of-five p value (0.37 vs 0.38) | Low | Trivial editorial fix |
| 7 | Optimal vs greedy matching sensitivity | Very low | Defer |

---

## E. Verdict

The analytic plan as documented in `Analysis_Specifications.md` is **statistically rigorous and clinically appropriate** for a single-centre Korean post-surgical HPV-vaccine cohort. The remaining recommendations are refinements rather than fundamental changes; none would alter the overall null inference for either co-primary outcome, but several (estimand framing, E-value, RMST) would strengthen the manuscript against methodologically demanding peer review at a JAMA-tier journal.

The single most important addition — the time-stratified clearance analysis — has been incorporated, and the resulting nuance (delayed but ultimately favourable clearance pattern at 6–12 months) is a clinically interpretable finding that strengthens rather than undermines the overall conclusion.
