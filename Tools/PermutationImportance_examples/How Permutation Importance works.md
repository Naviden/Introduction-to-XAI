# How Permutation Feature Importance Works

Permutation importance measures how much a model's score decreases when a single feature is randomly shuffled, breaking the relationship between the feature and the target.

## Algorithm

1. **Compute baseline score**: Evaluate the model on the (unshuffled) test set. Record the score s.
2. **For each feature j**:
   a. Randomly shuffle the values of feature j in the test set.
   b. Compute the model score s_j on the corrupted dataset.
   c. Feature importance = s - s_j.
3. **Repeat**: Shuffle multiple times (n_repeats) to get a distribution of importance scores.

## Formula

I_j = s - (1/K) Σ_{k=1}^{K} s_{j,k}

Where:
- s is the original model score
- s_{j,k} is the score after the k-th shuffle of feature j
- K is the number of repeats

## Key Properties

- **Model-agnostic**: Works with any model (no access to internals needed).
- **Computed on test data**: Reflects actual predictive importance, not training artifacts.
- **Handles interactions**: If a feature is only important through interaction, shuffling it still decreases performance.
- **Limitation**: Correlated features share importance — shuffling one may be partially compensated by the other.
