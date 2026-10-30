# How RuleFit works

**Question:** can a sparse combination of rules and optional linear terms predict the target or approximate another model?

RuleFit extracts conjunctions of feature conditions from a tree ensemble and fits a regularized model using those binary rules, optionally together with linear features:

$$F(x)=a_0+\sum_m a_m r_m(x)+\sum_j b_j\ell_j(x).$$

Each $r_m(x)$ is 0 or 1. Regularization selects and shrinks coefficients; rule generation and final coefficient fitting are different stages. A rule ensemble is not generally an OR of its rules. [Friedman and Popescu (2008)](https://arxiv.org/abs/0811.1679).

## Units and assumptions

The loss and link determine units: a regression score may be in target units; a logistic classification score is on the log-odds scale before its inverse link. A rule coefficient contributes only when that rule is active, alongside all other active terms and the intercept. If training uses black-box predictions as targets, evaluate **fidelity to that model** separately from predictive accuracy against observed labels.

## Checked example: evaluating fixed rules

These hand-set rules illustrate evaluation, not a fitted RuleFit result. Define R1 as balance > 4000 and R2 as age < 30 AND balance >= 3000. Use the arbitrary score R1 + 0.8 R2 and threshold 0.5; this score is not a probability. In this particular example either rule suffices, but that does not hold for arbitrary coefficients/intercepts.

<!-- checked-example: rulefit-rules -->
```python
customers = [(1, 25, 3000), (2, 47, 4500), (3, 35, 5000),
             (4, 52, 1500), (5, 20, 4000)]
print("customer R1 R2 score prediction")
for customer, age, balance in customers:
    r1 = int(balance > 4000)
    r2 = int(age < 30 and balance >= 3000)
    score = r1 + 0.8 * r2
    prediction = "Yes" if score >= 0.5 else "No"
    print(f"{customer} {r1} {r2} {score:.1f} {prediction}")
```
```text
customer R1 R2 score prediction
1 0 1 0.8 Yes
2 1 0 1.0 Yes
3 1 0 1.0 Yes
4 0 0 0.0 No
5 0 1 0.8 Yes
```
<!-- /checked-example -->

Customer 1 satisfies R2, including its equality boundary. Customer 5 does not satisfy R1 because that condition is strictly greater than 4000.

## Limitations and checks

Many overlapping rules can be hard to understand. Correlated rules can share or exchange coefficients across fits, and sparse rules can have low support. Report held-out performance, number/length/support of rules and their stability. For surrogates, include disagreements with the original model. Rules describe the fitted predictor; changing a feature to satisfy a rule is not evidence that an intervention will change a real outcome. See the [H2O notebook](RuleFit_example.ipynb) for a fitted example.
