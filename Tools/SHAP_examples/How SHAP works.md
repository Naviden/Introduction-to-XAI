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


---

### Example: Binary Classification

#### Scenario
We continue with our logistic regression model predicting diabetes likelihood, incorporating the original Shapley value calculations for a more mathematical and comprehensive explanation.

#### Original Model (Black Box)
Here’s the logistic regression equation again:
$$
\text{logit}(P(\text{Diabetes})) = -6 + 0.05 \times \text{Glucose Level} + 0.01 \times \text{BMI} + 0.02 \times \text{Age}
$$

#### Target Instance
Details for the individual (Bob):
- **Glucose Level**: 148
- **BMI**: 33.6
- **Age**: 50
- **Model Prediction Probability**: 0.7 (70% chance of having diabetes).

#### SHAP Values Calculation
SHAP values explain the output of the model by assigning each feature an importance value for a particular prediction.

##### Shapley Value Formula
The Shapley value for feature \( i \) is given by:
$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! (|N| - |S| - 1)!}{|N|!} [v(S \cup \{i\}) - v(S)]$$
where:
- \( N \) is the set of all features.
- \( S \) is a subset of features excluding \( i \).
- \( |S| \) is the number of features in subset \( S \).
- \( |N| \) is the total number of features.
- \( v(S) \) is the prediction value when only the features in subset \( S \) are used.
- \( \phi_i \) is the contribution of feature \( i \) to the prediction.

##### Applying Shapley Values
For our model, \( N = \{\text{Glucose Level}, \text{BMI}, \text{Age}\} \), we calculate the SHAP values by considering all subsets of features and their contributions:

1. **Baseline Prediction**:
   - \( \text{logit}(\text{Base}) = -6 \)

2. **Marginal Contributions**:
   - **Glucose Alone**, **BMI Alone**, **Age Alone**, and various combinations as detailed earlier are calculated.

3. **Computing SHAP Values** for each feature by averaging over all possible combinations of features:
   - **SHAP for Glucose**:
     - This value reflects the average change in the prediction when Glucose Level is added to all possible subsets of other features. Calculation involves multiple permutations where Glucose Level's presence or absence changes the outcome.
   - **SHAP for BMI** and **SHAP for Age**:
     - Similarly, calculated by systematically adding and removing these features from combinations and observing changes in predictions.

#### Detailed Calculations
We'll provide detailed computation results for one feature, considering its influence across all permutations:

- **For Glucose Level**:
  - Calculation involves combining contributions across all subsets, considering the factorial terms for weights, which account for the number of ways features can be arranged.