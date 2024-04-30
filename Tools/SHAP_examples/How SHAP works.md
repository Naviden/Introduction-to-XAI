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

The Shapley value for each feature is calculated by considering all possible subsets of features and the marginal contribution of the feature to the difference between the model prediction with and without the feature.


##### Shapley Value Formula
The Shapley value for feature $i$ is given by:
$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! (|N| - |S| - 1)!}{|N|!} [v(S \cup \{i\}) - v(S)]$$
where:
- $N$ is the set of all features.
- $S$ is a subset of features excluding $i$.
- $|S|$ is the number of features in subset $S$.
- $|N|$ is the total number of features.
- $v(S)$ is the prediction value when only the features in subset $S$ are used.
- $\phi_i$ is the contribution of feature $i$ to the prediction.

The above formula is a core component of SHAP and provides a way to fairly allocate the "payout" (or contribution) of each feature in a model's prediction. 

#### Formula Components:
1. **Subset $S$ Contribution**: The difference $[v(S \cup \{i\}) - v(S)]$ measures how much adding feature $i$ to the subset $S$ changes the prediction. If the difference is positive, $i$ is enhancing the prediction when added to $S$; if it's negative, $i$ is detracting from it.

2. **Weight of Each Contribution**: The weight $\frac{|S|! (|N| - |S| - 1)!}{|N|!}$ is a crucial part of the Shapley value calculation. It ensures each subset $S$ is weighted appropriately:
   - $|S|!$ accounts for all possible orders in which the features in $S$ can be arranged.
   - $(|N| - |S| - 1)!$ counts the permutations of the remaining features in $N$ after $S$ and $i$ have been chosen.
   - $|N|!$ is the total number of ways to arrange all features in $N$, serving as a normalizing factor to ensure the weights sum to 1.

#### Calculation Procedure:
The formula calculates the Shapley value by summing over all possible subsets $S$ of $N$ that exclude the feature $i$. For each subset, it computes the change in the prediction caused by adding $i$ to $S$, weighted by the probability of $S$ occurring in the order of features being added to a prediction model.

This approach ensures that the contribution of each feature is fairly evaluated in the context of all possible combinations of other features, reflecting the average marginal effect of including the feature in the model. This method attributes the contribution of each feature to the prediction output in a way that all contributions sum up to the total change in the prediction from the baseline (no features) to the full model.



### Step 4: Interpret SHAP Values

SHAP values can be interpreted directly as the impact of a feature on the model's prediction. A positive SHAP value indicates that the feature increases the prediction, while a negative value indicates a decrease. The magnitude of the SHAP value reflects the strength of the feature's impact.


---

### Example: Binary Classification

#### Scenario
Let's consider a logistic regression model predicting diabetes likelihood, incorporating the original Shapley value calculations for a more mathematical and comprehensive explanation.
#### Target Instance
Details for the individual (Bob):
- **Glucose Level**: 148
- **BMI**: 33.6
- **Age**: 50
- **Model Prediction Probability**: 0.7 (70% chance of having diabetes).



#### Setting
- **Features**: $N = \{\text{Glucose Level}, \text{BMI}, \text{Age}\}$
- **Prediction Model**: Logistic regression
  - $\text{logit}(P(\text{Diabetes})) = -6 + 0.05 \times \text{Glucose} + 0.01 \times \text{BMI} + 0.02 \times \text{Age}$

#### Steps to Calculate SHAP for Glucose Level
1. **Compute the prediction with no features (baseline)**:
   - $v(\emptyset) = -6$ (logit value with no features).

2. **Compute predictions for each subset including "Glucose Level"**:
   - **Only Glucose**: $v(\{\text{Glucose}\}) = -6 + 0.05 \times 148 = 1.4$
   - **Glucose + BMI**: $v(\{\text{Glucose, BMI}\}) = -6 + 0.05 \times 148 + 0.01 \times 33.6 = 1.436$
   - **Glucose + Age**: $v(\{\text{Glucose, Age}\}) = -6 + 0.05 \times 148 + 0.02 \times 50 = 1.4$
   - **Glucose + BMI + Age**: $v(\{\text{Glucose, BMI, Age}\}) = 1.476$ (full model).

3. **Compute predictions for each subset excluding "Glucose Level"**:
   - **Only BMI**: $v(\{\text{BMI}\}) = -6 + 0.01 \times 33.6 = -5.664$
   - **Only Age**: $v(\{\text{Age}\}) = -6 + 0.02 \times 50 = -5$
   - **BMI + Age**: $v(\{\text{BMI, Age}\}) = -6 + 0.01 \times 33.6 + 0.02 \times 50 = -5.664$

4. **Apply Shapley value formula**:
   - $\phi_{\text{Glucose}} = \sum_{S \subseteq N \setminus \{\text{Glucose}\}} \frac{|S|! (|N|-|S|-1)!}{|N|!} [v(S \cup \{\text{Glucose}\}) - v(S)]$

   For "Glucose Level":
   - Contribution when added alone: $[v(\{\text{Glucose}\}) - v(\emptyset)] = [1.4 - (-6)] = 7.4$
   - Contribution when added to BMI: $[v(\{\text{Glucose, BMI}\}) - v(\{\text{BMI}\})] = [1.436 - (-5.664)] = 7.1$
   - Contribution when added to Age: $[v(\{\text{Glucose, Age}\}) - v(\{\text{Age}\})] = [1.4 - (-5)] = 6.4$
   - Contribution when added to BMI + Age: $[v(\{\text{Glucose, BMI, Age}\}) - v(\{\text{BMI, Age}\})] = [1.476 - (-5.664)] = 7.14$

   Calculating the average of these contributions, adjusted for the number of subsets:
   - $\phi_{\text{Glucose}} = \frac{1}{3} (7.4) + \frac{1}{6} (7.1 + 6.4) + \frac{1}{3} (7.14) = 2.47 + 2.25 + 2.38 = 7.1$

### Explanation
The SHAP value for "Glucose Level" around 7.1 indicates a significant positive impact on increasing the probability of diabetes prediction, quantifying how much "Glucose Level" pushes the model's prediction toward the higher probability of diabetes compared to the baseline. This calculation takes into account the effect of "Glucose Level" across all possible combinations of features, weighted by the number of features included.