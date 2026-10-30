# How Grad-CAM works

**Question:** which spatial activations contribute positive evidence to a selected CNN output?

Select a differentiable scalar class score $y^c$ and a convolutional layer with activation maps $A^k$. Average the score's gradients across spatial positions and use those averages to weight the maps:

$$\alpha_k^c=\frac{1}{Z}\sum_{i,j}\frac{\partial y^c}{\partial A^k_{ij}},\qquad L^c=\operatorname{ReLU}\left(\sum_k\alpha_k^c A^k\right).$$

Upsample the resulting coarse map to overlay it on the input. No model retraining is required, but access to activations and gradients is necessary. [Selvaraju et al. (2017)](https://arxiv.org/abs/1610.02391).

## Units and assumptions

Record the exact target score and layer. The [VGG16 example](GradCAM_example.ipynb) uses a class logit. A normalized heatmap shows relative positive activation evidence for that image/layer/target; its colors are not calibrated probabilities or comparable absolute scores across images. ReLU discards negative evidence, and spatial resolution is limited by the selected feature maps.

## Limitations and checks

A visually plausible map does not prove faithful reasoning, object segmentation or a causal effect. Different layers/targets and input preprocessing can change the map. Gradients may vanish; handle an all-zero map without division by zero. Test class sensitivity, model randomization and input perturbations rather than judging quality only by visual appeal. Generic tabular models and models without suitable differentiable spatial features require a different approach or an explicitly adapted formulation.
