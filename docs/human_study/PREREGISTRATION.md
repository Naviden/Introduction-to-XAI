# Preregistration template

**Status: unfilled plan; no empirical results.** Complete and timestamp a frozen copy before confirmatory outcome inspection. Keep pilot changes distinguishable from later deviations.

## Question and hypotheses

- Target audience, task and deployment context: `[fill]`
- Primary question: `[Does format E change final decision accuracy relative to N?]`
- Primary estimand: `[mean within-participant E−N accuracy difference on the declared case pool]`
- Directional or two-sided hypothesis and rationale: `[fill before data collection]`
- Smallest practically meaningful effect: `[fill with task-specific justification]`
- Secondary hypotheses for time and the two conditional reliance measures: `[fill]`
- What result would count against the explanation's usefulness? `[fill]`

Use the definitions in [the measurement guide](../HUMAN_STUDY.md). Do not redefine the primary outcome after seeing results.

## Design and sampling

- Conditions, explanation format and fixed information in both conditions: `[fill]`
- Case blocks, matching variables and assignment scheme: `[fill]`
- Randomization seed/procedure and who can access allocations: `[fill]`
- Participant inclusion criteria and sampling source: `[fill]`
- Planned sample size and precision/power rationale: `[fill]`
- Trials per participant and expected eligible reliance trials: `[fill]`
- Recruitment/stopping rule, maximum sample/date and replacement policy: `[fill]`
- Model-error sampling proportion and intended generalization population: `[fill]`
- Pilot data exclusion from confirmatory analysis: `[fill]`

Avoid selecting an arbitrary sample size and calling it powered. If this is a feasibility pilot, label it as such and prioritize uncertainty/operational estimates over confirmatory significance claims.

## Data quality and exclusions

- Technical failure criteria: `[fill]`
- Duplicate or incomplete-session handling: `[fill]`
- Timing interruption, minimum/maximum duration and timeout policy: `[fill]`
- Missing initial/final responses: `[fill]`
- Prespecified attention/comprehension criteria and whether applied at trial or participant level: `[fill]`
- Treatment of zero eligible reliance denominators: `undefined; report denominator and missingness`
- Sensitivity analysis retaining otherwise valid excluded observations: `[fill]`

Do not exclude participants for low accuracy, surprising conclusions or disagreement with the model unless a justified criterion was declared independently of the outcome of interest.

## Analysis plan

- **Primary:** calculate each participant's E and N accuracy on eligible trials and the paired difference. Report the mean difference and a 95% participant-resampling bootstrap interval using `[iterations and seed]`. State the observed case-pool scope; this alone does not generalize to all possible cases.
- **Time:** summarize participant/condition median assisted-response time, with initial-response time separate. Predeclare any transformation and treatment of timeouts. Interpret jointly with accuracy.
- **Reliance:** report numerator and eligible denominator for each conditional measure by condition. Specify whether summaries are participant-weighted or pooled-trial rates and how repeated observations affect intervals; do not silently switch weighting.
- **Optional case generalization:** preregister a model with participant and case dependence, or an appropriate multiway resampling plan, if inference beyond the selected cases is intended. State model specification and convergence fallback in advance.
- Multiple comparisons and exploratory labels: `[fill]`
- Planned subgroup analyses and minimum eligible counts: `[fill]`
- Software versions and analysis-code commit: `[fill]`

The suggested paired analysis presumes the counterbalanced within-participant design. If using a between-participant design, replace it with the corresponding independent-group estimand and uncertainty procedure before collecting data.

## Governance and record

- Protocol/materials version and checksum: `[fill]`
- Institutional review determination/reference, if applicable: `[fill; do not invent approval]`
- Consent and withdrawal process: `[fill]`
- Access, retention, deletion and authorized sharing plan: `[fill]`
- Registration date/location and responsible investigator: `[fill]`
- Amendments: `[timestamp, what changed, why, whether any outcomes were already inspected]`
