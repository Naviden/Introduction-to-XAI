# SHapley Additive exPlanations (SHAP)

SHapley Additive exPlanations (SHAP) is a method for explaining individual predictions of machine learning models, based on the concept of Shapley values from cooperative game theory. SHAP values provide a way to understand the impact of each feature on the model's output, offering insights into the model's behavior at a granular level. This approach is model-agnostic, meaning it can be applied to any machine learning model.

## 1. Introduction to SHAP

SHAP leverages the idea of Shapley values, which originated in cooperative game theory to allocate profits (or costs) among players based on their contribution to the total profit (or cost). In the context of machine learning, SHAP values explain the prediction of an instance by quantifying the contribution of each feature to the prediction.

## 2. How SHAP Works

### Step 1: Select Your Instance for Explanation

Identify the instance $x$ you wish to explain, along with the output from the machine learning model for this instance.

### Step 2: Decompose the Model Prediction

For a given instance, SHAP decomposes the model prediction into the sum of contributions from each feature. The prediction can be represented as:

$$
f(x) = \phi_0 + \sum_{i=1}^{N} \phi_i
$$

where $f(x)$ is the model's prediction for the instance $x$, $N$ is the number of features, $\phi_0$ is the base value (the model's output when no features are present), and $\phi_i$ is the SHAP value or the contribution of feature $i$ to the prediction.

### Step 3: Calculate Shapley Values

The Shapley value for each feature is calculated by considering all possible subsets of features and the marginal contribution of the feature to the difference between the model prediction with and without the feature. Formally, the Shapley value for feature $i$ is given by:

$$
\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} [f(S \cup \{i\}) - f(S)]
$$

where $F$ is the set of all features, $S$ is a subset of features excluding feature $i$, $|S|$ is the number of features in subset $S$, and $|F|$ is the total number of features. $f(S \cup \{i\})$ is the model's prediction with the features in $S$ plus feature $i$, and $f(S)$ is the model's prediction with the features in $S$ alone.

### Step 4: Interpret SHAP Values

SHAP values can be interpreted directly as the impact of a feature on the model's prediction. A positive SHAP value indicates that the feature increases the prediction, while a negative value indicates a decrease. The magnitude of the SHAP value reflects the strength of the feature's impact.