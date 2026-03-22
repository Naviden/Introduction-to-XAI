# How Grad-CAM Works

Grad-CAM (Gradient-weighted Class Activation Mapping) explains CNN predictions by highlighting important image regions.

## Algorithm

1. **Forward pass**: Feed the image through the CNN and get the prediction for the target class.
2. **Backward pass**: Compute the gradient of the target class score with respect to the feature maps of the last convolutional layer.
3. **Global average pooling**: Average each gradient channel over its spatial dimensions to get importance weights α_k.
4. **Weighted combination**: Compute a weighted sum of the feature maps using these weights.
5. **ReLU activation**: Apply ReLU to keep only features with positive influence on the target class.

## Formula

α_k = (1/Z) Σ_i Σ_j ∂y^c / ∂A^k_ij

L_GradCAM = ReLU(Σ_k α_k · A^k)

Where:
- y^c is the score for class c (before softmax)
- A^k is the k-th feature map of the last convolutional layer
- α_k is the importance weight for feature map k
- Z is the number of spatial locations (width × height)

## Key Properties

- **Class-discriminative**: Produces different heatmaps for different classes.
- **Architecture-agnostic**: Works with any CNN that has convolutional layers.
- **No retraining needed**: Uses only the existing model's gradients.
