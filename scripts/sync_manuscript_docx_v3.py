"""
Final sync pass: tighten Abstract, Methods sensitivity para, Cohort B clearance
result, and subgroup paragraph in HPV_manuscript.docx to match the slimmed
Manuscript_Draft.md.
"""
from docx import Document

SRC = 'docs/HPV_manuscript.docx'
doc = Document(SRC)

REPLACEMENTS = [
    # Abstract — drop type-specific HPV-16/18 + quadrivalent sensitivity sentence
    (
        'Post-surgical vaccination was not associated with reduced lesion recurrence (hazard ratio 0.80, 95% CI 0.44–1.43; p = 0.45) or with accelerated clearance of pre-existing hr-HPV defined as two consecutive negative tests (hazard ratio 1.40, 0.92–2.11; p = 0.11). The clearance point estimate was directionally favourable but did not reach statistical significance, and type-specific clearance estimates for HPV 16 and HPV 18 were essentially null (1.05 and 0.84, both non-significant). A pre-specified piecewise time-stratified analysis of the clearance co-primary identified a significant 6–12 month window effect (hazard ratio 3.19, 95% CI 1.44–7.09; p = 0.004) consistent with the timing of HPV-vaccine antibody maturation. A vaccine-type interaction signal for any post-index hr-HPV detection in the full cohort (likelihood-ratio p = 0.037, driven by a quadrivalent-specific point estimate of 0.49 in 36 vaccinated women) did not persist when the analysis was restricted to the calendar window in which all three products co-existed (2016–2018, p = 0.105) and was further attenuated by landmark sensitivity, suggesting calendar-period and surgery-to-vaccination-timing confounding rather than a quadrivalent-specific mechanism. HPV vaccination did not increase chronic-disease risk over a median follow-up of approximately five years.',
        'Post-surgical vaccination was not associated with reduced lesion recurrence (hazard ratio 0.80, 95% CI 0.44–1.43; p = 0.45) or with accelerated overall clearance of pre-existing hr-HPV defined as two consecutive negative tests (hazard ratio 1.40, 0.92–2.11; p = 0.11). The clearance point estimate was directionally favourable but did not reach statistical significance. A pre-specified piecewise time-stratified analysis of the clearance co-primary identified a significant 6–12 month window effect (hazard ratio 3.19, 95% CI 1.44–7.09; p = 0.004) consistent with the timing of HPV-vaccine antibody maturation. HPV vaccination did not increase chronic-disease risk over a median follow-up of approximately five years.'
    ),
    # Cohort B clearance prose — drop type-specific HPV-16/18 sentence
    (
        'The point estimate was directionally favourable but did not reach statistical significance. Type-specific clearance estimates were essentially null (HPV 16: 1.05, 0.74 to 1.49, p = 0.79; HPV 18: 0.84, 0.56 to 1.26, p = 0.40). The Schoenfeld residual rank test indicated',
        'The point estimate was directionally favourable but did not reach statistical significance. The Schoenfeld residual rank test indicated'
    ),
    # Methods sensitivity additions paragraph (P39 of Manuscript_Draft markdown, similar block)
    (
        'Additional analyses (post-index hr-HPV detection, landmark variants, novel-type acquisition, type-specific clearance for HPV 16 and 18, vaccine-type × calendar-period interaction, restricted-follow-up and unadjusted variants, and prescription-code exposure validation) are reported as supplementary material only.',
        'The legacy post-index hr-HPV detection endpoint (which conflates persistence and new acquisition) is retained as a sensitivity row in the cluster-robust hazard ratio table (Supplementary Table S6) and in the vaccine-type interaction table (Supplementary Table S5) for transparency, but is not interpreted as a co-primary outcome.'
    ),
    # Subgroup paragraph (P52) tightening
    (
        'The likelihood-ratio test for vaccine-type heterogeneity was non-significant for both co-primary outcomes (lesion recurrence χ² = 0.64 on 2 d.f., p = 0.72; hr-HPV clearance χ² = 3.07 on 2 d.f., p = 0.22), and the test for age × vaccination interaction was likewise non-significant for both outcomes (lesion recurrence p = 0.07; hr-HPV clearance p = 0.37). The type-specific clearance hazard ratios were inconsistent in direction (Gardasil 9 0.72, 95% CI 0.34 to 1.51; Cervarix 1.43, 0.88 to 2.34; quadrivalent Gardasil 1.55, 0.73 to 3.31). The post-index hr-HPV detection sensitivity outcome showed nominal heterogeneity (χ² = 11.27, p = 0.004), driven by a quadrivalent-specific point estimate of 0.40 (0.21 to 0.76); this signal was absent in the co-primary clearance outcome and was no longer significant when the analysis was restricted to the 2016–2018 calendar window in which all three products co-existed (p = 0.105; calendar-restricted clearance LRT p = 0.24). The surgery-to-vaccination interval differed by product (median 71 days for nine-valent, 134 days for quadrivalent, and 138 days for Cervarix). The vaccine-type and age-stratified subgroup patterns are reported as hypothesis-generating; a separate exploratory finding identified by a grid search over many overlapping age ranges and follow-up windows (a 2-year-censored estimate for lesion recurrence among women aged 30 to 52 years, hazard ratio 0.23, 95% CI 0.06 to 0.96, p = 0.04) is acknowledged in the limitations section as a single nominally significant cell among approximately 6,900 combinations tested, is not displayed in the main figure, and is not used to support any inference.',
        'The likelihood-ratio test for vaccine-type heterogeneity was non-significant for both co-primary outcomes (lesion recurrence χ² = 0.64, p = 0.72; hr-HPV clearance χ² = 3.07, p = 0.22), and the age × vaccination interaction was likewise non-significant (lesion recurrence p = 0.07; hr-HPV clearance p = 0.37; Supplementary Table S5). A nominal quadrivalent-specific signal appeared in the legacy post-index hr-HPV detection sensitivity row of the same vaccine-type interaction model (HR 0.40, 95% CI 0.21–0.76; LRT p = 0.004); we interpret this as residual calendar-period and surgery-to-vaccination-timing confounding rather than a vaccine-type effect (Discussion). An exploratory grid search across overlapping age ranges and follow-up windows identified a single nominally significant cell for lesion recurrence (women aged 30–52 years, 2-year follow-up: HR 0.23, 95% CI 0.06–0.96) that did not survive any reasonable adjustment for the ≈6,900 combinations tested and is not used to support inference (Supplementary Table S3).'
    ),
]

n = 0
for old, new in REPLACEMENTS:
    for p in doc.paragraphs:
        if old in p.text:
            # rewrite paragraph (preserving the first run)
            new_text = p.text.replace(old, new)
            for r in list(p.runs):
                r.text = ''
            p.runs[0].text = new_text
            n += 1
            break  # only one paragraph should match

doc.save(SRC)
print(f'Paragraphs rewritten: {n} of {len(REPLACEMENTS)}')
