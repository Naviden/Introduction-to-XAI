
# Explainable AI (XAI): Theory, examples, ...
## Overview

This repository offers practical examples and educational resources to help you understand Explainable AI (XAI). 
It includes Jupyter notebooks and Python scripts that demonstrate the use of various XAI frameworks, such as 
LIME, SHAP, Anchors, Grad-CAM, Integrated Gradients, ALE plots, DiCE, and more.
The aim is to provide a hands-on approach to interpreting machine learning model predictions and to shed light
on the decision-making processes of complex algorithms.

## Contents

- `Theory/` - Jupyter notebooks covering XAI fundamentals: white-box vs. black-box models, surrogate models, global vs. local explanations, and explanation types (plain-fact, counterfactual, contrastive).
- `Tools/` - Hands-on examples for various XAI frameworks: LIME (tabular, text, image), SHAP (tabular, summary plots, survival models, text/image), Anchors, RuleFit, MERLIN, PDP/ICE plots, Grad-CAM, Integrated Gradients, ALE plots, Permutation Importance, Attention Visualization, DiCE (counterfactual explanations), and a Feature Importance Comparison.
- `Practice/` - Practice notebooks for students to apply XAI techniques.

## Getting Started

To explore the examples in this repository, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Naviden/Introduction-to-XAI.git
   ```

2. **Create a virtual environment with Python 3.11 and install dependencies:**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

3. **Download the spacy model (needed for the Anchors notebook):**
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Open the Jupyter notebooks:**
   ```bash
   jupyter notebook
   ```

## Contributing

Contributions are welcome! If you'd like to add new examples, enhance existing ones, or suggest additional XAI frameworks to include, please submit a pull request or open an issue.
