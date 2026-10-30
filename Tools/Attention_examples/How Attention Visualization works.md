# How attention visualization works

**Question:** how does a selected attention head mix token representations at a particular layer?

Scaled dot-product attention computes

$$A=\operatorname{softmax}(QK^T/\sqrt{d_k}+M),\qquad\operatorname{Attention}=AV,$$

where $M$ masks disallowed positions. In evaluation mode, each row of $A$ is a normalized distribution over permitted key positions for one query. The attended values, output projections, other heads, residual connections and later layers also affect the prediction. [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762).

## Units and assumptions

Weights are dimensionless, nonnegative mixing coefficients, not signed contributions to a selected output class. State the model, layer/head, query/key orientation, tokenization and masking. A row sums to one before attention dropout; returned training-mode weights may differ. Averaging heads or layers is a visualization choice that can conceal distinct behavior.

## Limitations and interpretation

Inspecting a heatmap can generate hypotheses about representation use. High attention alone does not show that a token causes the final decision or is the most important input. Evidence about attention's explanatory usefulness depends on the task and the test: compare [Jain and Wallace (2019)](https://aclanthology.org/N19-1357/) with [Wiegreffe and Pinter (2019)](https://aclanthology.org/D19-1002/).

For prediction-level claims, test ablations/interventions and appropriate attribution methods, with their own assumptions. Neither agreement with gradients nor an attractive pattern is a universal validation. The [notebook](Attention_visualization.ipynb) visualizes encoder attention; it does not demonstrate faithful natural-language rationales or generative-model reasoning.
