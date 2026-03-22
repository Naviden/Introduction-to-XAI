# How DiCE (Diverse Counterfactual Explanations) Works

DiCE generates "what-if" scenarios: minimal changes to an input that would change the model's prediction.

## Algorithm

DiCE optimizes for three objectives simultaneously:

1. **Validity**: The counterfactual must have the desired prediction.
2. **Proximity**: The counterfactual should be close to the original input (minimal changes).
3. **Diversity**: Multiple counterfactuals should be different from each other.

## Optimization

The loss function balances:

```
L = λ₁ · yloss(f(cf), y_desired) + λ₂ · dist(cf, x) - λ₃ · diversity(cf₁, cf₂, ..., cf_k)
```

Where:
- **yloss**: ensures the counterfactual has the desired prediction
- **dist**: penalizes large changes from the original input
- **diversity**: encourages diverse counterfactuals (uses determinantal point processes or simple pairwise distance)

## Methods

DiCE supports multiple generation methods:
- **Random**: Sample random perturbations, keep valid ones.
- **Genetic**: Evolutionary algorithm to evolve counterfactuals.
- **KD-tree**: Use nearest neighbors from the opposite class in the training data.

## Key Properties

- **Actionable**: Constraints can limit which features change and their allowed ranges.
- **Model-agnostic**: Works with any classifier.
- **User-friendly**: "Change X to Y" is intuitive for non-technical users.
