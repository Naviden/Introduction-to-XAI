# How Anchors work

**Question:** under which feature conditions does a classifier usually keep this prediction when other inputs are sampled from a specified distribution?

Anchors are rule explanations from Ribeiro, Singh and Guestrin's [Anchors: High-Precision Model-Agnostic Explanations (2018)](https://homes.cs.washington.edu/~marcotcr/aaai18.pdf). The historical filename is retained for links; the method is called **Anchors**.

For a rule $A$ satisfied by the query $x$, define

$$\operatorname{prec}(A)=P_{z\sim D}[f(z)=f(x)\mid A(z)=1],\qquad\operatorname{cov}(A)=P_{z\sim D}[A(z)=1].$$

The aim is high coverage subject to a required precision $\tau$. Precision is estimated by sampling; the search uses statistical confidence bounds and beam search rather than an exhaustive proof. Its target is a precision condition with confidence at least $1-\delta$ under the sampling assumptions. Maximum rule length and sampling budgets can prevent a satisfactory anchor from being found.

## Units and assumptions

Precision and coverage are proportions under **D**, the chosen perturbation distribution. They are not test accuracy, probabilities of a real-world outcome, or guarantees over every possible input. Record D, discretization, requested precision, confidence settings, sample budget and observed estimates. Changing D can change which rule is useful.

## Example and limitations

Suppose a toy classifier approves exactly when credit score > 700 AND income > 50000. The two-condition rule is sufficient for this deterministic toy function. Holding the conditions does not freeze feature values: credit score and income may both vary while remaining above their thresholds. A learned real-model anchor usually provides a probabilistic approximation, not this logical guarantee.

Short rules can cover very few cases, and perturbations may be unrealistic. Check both precision and coverage, inspect sampled failures and report unmet thresholds. The [historically named notebook](Anchor_tabular.ipynb) covers text; the [shared comparison](../MethodComparison_examples/Shared_Tabular_Case_Comparison.ipynb) includes tabular Anchors.
