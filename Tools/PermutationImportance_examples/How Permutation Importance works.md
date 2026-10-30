# How permutation feature importance works

**Question:** how much does a fixed model's score change when one feature's association with the other data is disrupted?

Compute score $s$ on a reference dataset, shuffle feature $j$, recompute $s_{jk}$, and repeat for $K$ shuffles:

$$I_j=s-\frac{1}{K}\sum_{k=1}^{K}s_{jk}.$$

Use a score where larger means better; scikit-learn negates loss metrics to follow this convention. The model is not refitted. Shuffling breaks the feature's observed association with the target **and with other features**. [Scikit-learn documentation](https://scikit-learn.org/stable/modules/permutation_importance.html); the random-forest permutation approach originates in [Breiman (2001)](https://doi.org/10.1023/A:1010933404324).

## Units and assumptions

Importance is a difference in the selected score: accuracy, ROC AUC, R² or a signed loss score. State the evaluation population, metric and number of shuffles. Held-out data evaluates reliance for generalization; training data instead measures reliance on training cases. This choice is not fixed by the algorithm.

## Limitations and checks

Held-out permutation avoids the split-selection bias of impurity importance, but does not reveal a universal “true feature importance.” Correlated substitutes can mask reliance; shuffling can also create unrealistic feature combinations. Grouped permutations ask about joint reliance and answer a different question. An interaction can affect permutation importance, but the statistic neither isolates that interaction nor guarantees a positive drop.

Assess model quality first, report shuffle variability and inspect negative values. Shuffle standard deviation describes randomness conditional on the fitted model and sample; it is not a population confidence interval. Resampling/refitting is needed to study those additional uncertainty sources. Negative importance can occur through sampling variation or harmful model reliance. Removing a feature and retraining is **drop-column importance**, a separate procedure. See the [notebook](PermutationImportance_example.ipynb).
