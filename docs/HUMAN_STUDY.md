# Evaluate whether explanations help people

**Status: study design materials only. No participants have been recruited, no responses collected, and no study results are claimed here.**

The notebooks test mathematical and predictive properties. A human-centered evaluation asks a different question: does an explanation help a particular audience perform a specified task? Choose the audience and task before choosing a preferred chart. This distinction follows the evaluation agenda of [Doshi-Velez and Kim](https://arxiv.org/abs/1702.08608).

Use these editable materials:

- [Task protocol and trial log](human_study/PROTOCOL.md)
- [Preregistration template](human_study/PREREGISTRATION.md)
- [Participant information and consent draft](human_study/PARTICIPANT_INFORMATION.md)
- [Report template](human_study/REPORT_TEMPLATE.md)

## A concrete comparison

Compare the **same model recommendation and confidence display**, with and without one fixed explanation format. Hold wording, decision options, input data and feedback constant. Participants first answer without model assistance, then see the assigned display and make a final decision. An initial response is necessary to distinguish useful advice from harmful switching.

Prepare two matched, nonoverlapping case sets and counterbalance both display condition and order. Each person sees every case at most once. Include model errors as well as correct recommendations; otherwise a design cannot evaluate inappropriate reliance. If errors are intentionally oversampled, report that case mix and limit conclusions accordingly. The [protocol](human_study/PROTOCOL.md) gives the four assignment sequences.

In one prominent study, explanations increased acceptance of recommendations without improving complementary team performance; persuasion should therefore not be treated as the goal. See [Bansal et al.](https://arxiv.org/abs/2006.14779).

## Predeclare the measures

| Measure | Operational definition | Interpretation |
|---|---|---|
| **Primary: final decision accuracy** | Fraction of eligible cases with final answer equal to the established reference answer, summarized per participant and condition | Measures task performance, not agreement with the AI |
| Decision time | Time from assistance-display availability to final answer; report initial-response time separately | Faster is not better when accuracy worsens; handle interruptions as preregistered |
| Appropriate use of correct advice | Among trials with initial human answer wrong and AI answer correct, fraction ending with the correct answer | Report the eligible numerator/denominator, not just a percentage |
| Appropriate rejection of wrong advice | Among trials with initial human answer correct and AI answer wrong, fraction retaining the correct final answer | Measures successful resistance to incorrect advice |
| Optional comprehension | Accuracy on predefined questions about the explanation's target, units, scope or limitations | Tests understanding, separate from liking or trust ratings |

The two conditional reliance measures adapt the two-dimensional approach in [Schemmer et al.](https://arxiv.org/abs/2302.02187). A zero eligible denominator is **undefined**, not a zero rate. Cases where both human and model are wrong need separate reporting. Agreement, self-reported trust and aesthetic preference are not substitutes for correctness or appropriate reliance.

For the primary analysis, estimate the difference between explanation and no-explanation conditions with participant dependence respected. The templates propose a participant-level paired contrast and interval; trial-level extensions must also account for repeated participants and cases. Record the planned sample size, exclusions, missing-data rules and stopping rule before outcome inspection. Treat additional comparisons as exploratory or predeclare a multiple-comparison plan.

## How to use the materials

1. Specify a low-risk task and a target population; verify case labels and explanation outputs independently.
2. Adapt the protocol, participant-facing information and data-handling fields. Determine applicable institutional review requirements before a real study; this repository does not claim approval.
3. Pilot the interface and instructions with data kept separate from the confirmatory analysis. Revise the materials, then freeze the protocol and analysis plan.
4. After authorized data collection, report the actual sample, exclusions, uncertainty, negative findings and deviations. Do not fill the report with invented demonstrations presented as empirical evidence.

Keep identifying information and individual participant responses out of this public teaching repository. Publish only material whose sharing is authorized, with an explicit data-availability statement.
