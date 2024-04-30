# Local Interpretable Model-agnostic Explanations (LIME)

Local Interpretable Model-agnostic Explanations (LIME) is a technique designed to help us understand the predictions of any machine learning model by approximating the model locally around the prediction. LIME is particularly useful for explaining individual predictions of complex, black-box models such as deep neural networks or ensemble methods. Here's a detailed breakdown, including some essential formulas.

## 1. Introduction

LIME generates explanations by creating a simple model (like a linear regression or decision tree) that approximates the behavior of the complex model locally. This is done around the instance (data point) you're interested in explaining. The idea is that while the global model is complex and possibly non-linear, its behavior near a specific instance can be approximated by a simpler, interpretable model.

## 2. How LIME Works

### Step 1: Select Your Instance for Explanation

Suppose you have a complex model $f$ and an instance $x$ whose prediction you want to explain. $x$ can be anything from a row of tabular data to an image or a piece of text, depending on the model's input.

### Step 2: Perturb the Dataset

Generate a new dataset of perturbed samples around $x$. This involves creating variations of $x$ by tweaking the input features slightly. For tabular data, this might involve changing the values of some features. For images, this could mean masking parts of the image. For text, this could involve removing words or phrases.

### Step 3: Use the Complex Model to Predict Outcomes of Perturbed Samples

Use the original complex model $f$ to make predictions on this new dataset of perturbed samples. This gives you a set of outcomes that reflect how changes in the input influence the prediction.

### Step 4: Weight the Perturbed Samples

Assign weights to the perturbed samples based on their similarity to the original instance $x$. A common way to do this is by using an exponential kernel:

$$
w = exp\left(-\frac{d(x, x')^2}{\sigma^2}\right)
$$

where $d(x, x')$ is a distance measure between the original instance $x$ and a perturbed instance $x'$, and $\sigma$ controls the width of the kernel, determining how the similarity decreases as the distance increases.

### Step 5: Train a Simple Model

Train a simple, interpretable model (like linear regression) on the dataset of perturbed samples, using the outcomes predicted by the complex model as the target variable and the perturbed sample features as the input. Use the weights $w$ from the previous step as sample weights during training, ensuring that perturbed samples closer to $x$ have a greater influence on the model.

The formula for the linear regression model, for instance, would aim to minimize the weighted least squares:

$$
\min_{\theta} \sum w_i (f(x_i) - \theta^T x_i)^2
$$

where $f(x_i)$ is the prediction of the complex model for the perturbed instance $x_i$, $\theta^T x_i$ is the prediction of the simple linear model, and $w_i$ is the weight for instance $x_i$.

### Step 6: Interpret the Simple Model

The parameters (like coefficients in linear regression) of the simple model serve as explanations for the prediction at $x$. For example, in a linear model, the coefficients indicate the direction and magnitude of the influence of each feature on the prediction.





### Example: Binary Classification

#### Scenario
We have a model trained to predict if individuals earn more than $50k a year. The features used in the model are:
- **Age**: Continuous variable.
- **Education Level**: Categorical variable (High School, Bachelor's, Master's).

#### Original Model (Black Box)
Let's assume our decision tree has learned the following (simplified for explanation):
- If age > 30 and has at least a Bachelor's degree, predict > $50k.
- Otherwise, predict <= $50k.

#### Target Instance
We want to explain the model's prediction for Alice:
- **Age**: 35
- **Education Level**: Bachelor's
- The model predicts: > $50k (positive class).

#### Step 1: Data Perturbation
We generate new samples around Alice's features by slightly modifying her age and education:
```
| Age | Education     | Model Prediction |
|-----|---------------|------------------|
| 34  | Bachelor's    | > $50k           |
| 32  | High School   | <= $50k          |
| 37  | Master's      | > $50k           |
| 29  | Bachelor's    | <= $50k          |
| 36  | High School   | <= $50k          |
| 33  | Bachelor's    | > $50k           |
```

#### Step 2: Weight Calculation
Weights are assigned based on proximity to Alice's actual features (Age: 35, Education: Bachelor’s), using an exponential kernel with $\sigma = 3$ for simplicity:

$Weights = exp(-(distance^2) / 9)$



Distances (simplified calculation considering categorical as a change/no-change and numerical distance for age):
```
| Perturbed Sample | Distance | Weight   |
|------------------|----------|----------|
| (34, Bachelor's) | 1        | 0.90     |
| (32, High School)| 4        | 0.45     |
| (37, Master's)   | 2.5      | 0.72     |
| (29, Bachelor's) | 6        | 0.25     |
| (36, High School)| 3.5      | 0.57     |
| (33, Bachelor's) | 2        | 0.82     |
```

#### Step 3: Train Local Interpretable Model
We train a simple linear regression model using the perturbed data, with the outcome variable being the binary prediction (1 for > $50k, 0 for <= $50k), and weights as calculated:

```
Model: Predicted = 0.1 * Age + 0.5 * Education_Bachelor’s + 0.4 * Education_Master’s
```

#### Step 4: Explanation Generation
The coefficients indicate the importance of each feature:
- **Age Coefficient**: 0.1 — Suggesting that an increase in age slightly increases the chance of earning > $50k.
- **Education Coefficients**: Bachelor's (0.5) and Master's (0.4) — Indicating that having at least a Bachelor's degree significantly influences higher earnings predictions.