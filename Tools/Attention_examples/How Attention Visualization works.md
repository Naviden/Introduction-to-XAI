# How Attention Visualization Works

Attention visualization displays the self-attention weights from transformer models to show which tokens influence each other during processing.

## Mechanism

In a transformer, self-attention computes:

Attention(Q, K, V) = softmax(QK^T / √d_k) V

The softmax(QK^T / √d_k) matrix contains the attention weights — a probability distribution showing how much each token attends to every other token.

## What Attention Reveals

- **Syntactic patterns**: Some heads learn subject-verb agreement or dependency parsing.
- **Positional patterns**: Some heads attend primarily to adjacent or nearby tokens.
- **Semantic patterns**: Some heads attend to semantically related tokens.

## Visualization Approaches

1. **Single-head heatmap**: A matrix where entry (i,j) shows how much token i attends to token j.
2. **Multi-head comparison**: Side-by-side heatmaps for different heads, showing each captures different relationships.
3. **Aggregated attention**: Average or max across heads/layers for a global view.

## Limitations

- Attention ≠ explanation: High attention doesn't necessarily mean high importance for the final prediction.
- Gradient-based methods (like Integrated Gradients) provide more principled attributions.
- Attention patterns can be misleading — attending to a token doesn't mean using it for the output.
