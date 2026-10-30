# Scope and next extensions

This repository aims to let readers understand, choose, run, evaluate, adapt and
cite representative XAI methods. Inclusion is based on the distinct question a
method answers, a maintained implementation, an executable example and a way to
evaluate its limitations—not on collecting every named variation.

## Core foundation implemented

- Checked SHAP/LIME/RuleFit arithmetic and corrected method assumptions/units.
- Lightweight starter and resolved tabular environments.
- Method-selection guide, evaluation protocol and contribution contract.
- Intrinsic comparison: logistic regression, small tree and additive spline GAM.
- Repeated synthetic evaluation, background sensitivity and a shortcut failure case.
- Causal-recourse example separating feature edits from interventions.
- A mixed-data adaptation workflow and a shared image attribution comparison.
- Durable executed examples, source/helper hashes and environment manifests.
- MIT original code, CC BY 4.0 original teaching content and citation metadata.

## Next core extensions

- EBM alongside the spline GAM; DeepLIFT/LRP alongside existing neural baselines.
- A shared text-classification comparison with explicit token/masking semantics.
- Broader benchmark tasks (including regression), more splits/cases and measured
  resource use. The current small study illustrates a protocol, not a leaderboard.
- Immutable revisions and checksums for all legacy external data/model downloads;
  fuller third-party notice packaging and verified asset reuse records.
- Complete resolved profiles for optional neural/Java/MERLIN stacks.
- More exercises, separate instructor solutions, and human-study materials.

## Research extensions

Concept explanations (TCAV and concept bottlenecks), prototypes/criticisms,
training-data influence, GNNs, temporal models, generative model attribution and
mechanistic interpretability. These remain outside the current implementation.

Each addition should meet [the contribution checklist](../CONTRIBUTING.md): a
precise question, assumptions, units, reproducible code, meaningful checks,
failure case, interpretation and primary sources. Report model quality separately
from explanation fidelity or human utility.
