# Protocol template: explanation versus no explanation

**Template — not an executed study.** Replace every bracketed field before use. Pair with the [preregistration](PREREGISTRATION.md), [participant information](PARTICIPANT_INFORMATION.md) and [report](REPORT_TEMPLATE.md).

## Fixed study definition

- Study identifier and protocol version: `[fill]`
- Audience and task expertise: `[fill]`
- Specific task: `[e.g., classify a synthetic item using stated inspection criteria]`
- Intended benefit of the explanation: `[detect wrong advice, understand a model limitation, or another measurable task]`
- Model and preprocessing version/hash: `[fill]`
- Explanation method, parameters, target output, reference and format: `[fill]`
- Interface/screenshot version: `[fill]`
- Known answer source and independent verification procedure: `[fill]`
- Case selection, difficulty matching and model-error proportion: `[fill]`

Use a task with an assessable reference answer. An explanation is a representation of model behavior, not the answer key. Keep the recommendation, displayed confidence and interaction controls identical across conditions.

## Conditions and counterbalancing

- **E:** input + model recommendation/confidence + explanation.
- **N:** input + the same recommendation/confidence; explanation area absent or neutral with the layout choice fixed in advance.

Create matched, nonoverlapping blocks A and B. Assign participants in balanced random blocks to these sequences:

| Sequence | First block | Second block |
|---|---|---|
| 1 | A with E | B with N |
| 2 | B with N | A with E |
| 3 | A with N | B with E |
| 4 | B with E | A with N |

Randomize case order within a block using recorded seeds. Do not show the same case to the same person in both conditions. The sequence balances condition order and case-block allocation across participants; it does not eliminate learning or carryover, which must be discussed. If carryover is unsuitable for the task, preregister a between-participant design instead.

## Session script

1. Present the study information and obtain the adapted consent response.
2. Explain the task, available information and that the model can be wrong. Do not imply that explanation detail certifies correctness.
3. Present `[number]` practice cases outside the analysis pool with the same instruction for every condition.
4. On each trial, show the input only and record the initial answer and initial-response duration.
5. Reveal the assigned assistance display; start the second timer only once it is fully available.
6. Record the final answer, assistance-response duration and optional confidence/comprehension answers.
7. Give feedback according to the preregistered policy: `[after each trial / after all trials / none]`. Use the same policy in both conditions.
8. Offer a break between blocks; record interruptions without silently deleting poor-performance trials.
9. End with the debriefing and data-withdrawal instructions specified in the participant material.

Neutral trial prompt: “Make your initial choice using the displayed input. You will then see an automated recommendation that may be correct or incorrect. Make your final choice using whichever information you consider useful.”

## Trial log schema

Store one row per trial, using a nonidentifying participant code. Store consent/contact records separately where applicable.

| Field | Meaning |
|---|---|
| `participant_code`, `sequence`, `block`, `trial_index` | Repeated-measure identifiers and assigned order |
| `case_id`, `condition`, `case_set_version` | Exact input and display assignment |
| `model_version`, `explanation_version` | Reproducible assistance source |
| `reference_answer`, `model_answer`, `model_correct` | Verified task label and model output |
| `initial_answer`, `final_answer` | Independent response and assisted decision |
| `initial_ms`, `assisted_ms` | Separate measured durations |
| `interruption_flag`, `technical_error_flag` | Prespecified timing/validity diagnostics |
| `comprehension_answer` | Optional response using a fixed scoring key |
| `exclusion_code` | Preregistered exclusion reason; never inferred solely from disagreeing with the AI |

Optional demographic or expertise fields must have a stated analytic need and suitable collection/sharing arrangements. This template does not require names, health information or other sensitive participant data.

## Preflight checks

- Verify ground-truth labels without reference to the explanation output.
- Confirm that a given case has identical model outputs under E and N.
- Confirm assignment coverage, timing behavior and that no participant repeats a case.
- Check that each case set includes enough correct and incorrect model recommendations for the planned analyses.
- Confirm that malformed or missing observations remain traceable; do not silently fill them with zeros.

These checks validate materials and logging. They do not constitute evidence that the explanation improves human performance.
