# How ALE (Accumulated Local Effects) Works

ALE plots show the isolated effect of a feature on the prediction, without being biased by feature correlations (a known issue with Partial Dependence Plots).

## Algorithm

1. **Divide feature range into intervals**: Split feature x_j into K small intervals (bins).
2. **Compute local effects**: For each interval, compute the average difference in predictions when the feature moves from the lower to the upper bound of the interval, using only the data points that fall within that interval.
3. **Accumulate**: Sum up the local effects from left to right across intervals.
4. **Center**: Subtract the mean so the ALE effect averages to zero.

## Formula

ALE(x_j) = Σ_{k=1}^{k_j} (1/n_k) Σ_{i: x_ij ∈ N_k} [f(z_k, x_i\j) - f(z_{k-1}, x_i\j)] - constant

Where:
- N_k is the k-th interval
- n_k is the number of samples in interval k
- z_k and z_{k-1} are the upper and lower bounds of interval k
- x_i\j are all features except x_j (kept at their observed values)

## Why ALE > PDP for Correlated Features

PDP marginalizes over all feature values, creating impossible combinations (e.g., 10 rooms but low income). ALE only uses nearby observed values, avoiding this issue.
