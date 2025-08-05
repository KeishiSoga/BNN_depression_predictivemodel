# Bayesian Neural Network for Early Depression Prediction in Older Adults

This repository contains the implementation of a Bayesian Neural Network (BNN) model for early depression prediction in older adults, with comprehensive analysis of humor styles, communication skills, and social physical activity experiences.

## Overview

This project develops a probabilistic machine learning approach using Bayesian Neural Networks to predict depression risk in older adults based on multidimensional behavioral and psychological factors. The model leverages uncertainty quantification to provide reliable predictions with confidence intervals.

## Research Background

Depression in older adults is a significant public health concern that often goes undetected in early stages. This study investigates the relationship between various psychological and behavioral factors and depression risk, focusing on:

- **Humor Styles**: Different types of humor usage and their psychological impact
- **Communication Skills**: Social communication abilities and interpersonal skills
- **Social Exercise Experiences**: Physical activity in social contexts and its mental health benefits

## Methodology

### Model Architecture
- **3-layer Bayesian Neural Network** with LeakyReLU activation
- **MCMC sampling** using No-U-Turn Sampler (NUTS) for posterior inference
- **Variational inference** for uncertainty quantification

### Data Processing
- **Target Variable**: K6 psychological distress scale (binary classification: <9 vs ≥9)
- **Features**: 
  - Numeric features: Age, exercise metrics, humor scores, communication skills
  - Categorical features: Gender, social exercise experiences
- **Preprocessing**: Min-Max scaling for numeric features

### Key Features
- **Uncertainty Quantification**: Provides prediction confidence intervals
- **Feature Importance**: SHAP analysis for model interpretability
- **Threshold Optimization**: Optimized classification threshold for best F1-score
- **Comprehensive Evaluation**: Multiple metrics including precision, recall, F1-score

## Dependencies

```python
# Core ML and statistical libraries
numpy
jax
jaxlib
numpyro
optax

# Data processing and evaluation
pandas
scikit-learn
scipy

# Visualization
matplotlib
seaborn
arviz

# Model interpretation
shap

# Additional utilities
watermark (optional, for version tracking)
```

## Installation

```bash
pip install numpy jax jaxlib numpyro optax pandas scikit-learn scipy matplotlib seaborn arviz shap
```

## Usage

1. **Prepare your data**: Ensure your CSV file contains all required features as specified in the feature lists within the code.

2. **Run the model**:
```bash
python BNN_depression_predictivemodel.py
```

3. **Key outputs**:
   - Model performance metrics
   - Confusion matrices
   - Prediction probability distributions
   - SHAP feature importance plots
   - Saved model artifacts (NetCDF format)

## Model Performance

The model provides:
- **Probabilistic predictions** with uncertainty quantification
- **Optimized threshold** for balanced precision-recall trade-off
- **Feature interpretability** through SHAP values
- **Credible intervals** for prediction confidence

## Results Structure

- `idata_EX_K6_5_2_bina_9_5_19.nc`: Saved posterior samples in NetCDF format
- Visualization outputs: Various plots showing model performance and feature importance
- Console outputs: Detailed metrics and model summaries

## Key Features of the Implementation

### Bayesian Approach Benefits
- **Uncertainty Quantification**: Provides confidence in predictions
- **Robust to Overfitting**: Regularization through prior distributions
- **Interpretable Results**: Posterior distributions for all parameters

### Model Interpretability
- **SHAP Analysis**: Feature importance and contribution analysis
- **Threshold Optimization**: Data-driven classification threshold selection
- **Comprehensive Metrics**: Multiple evaluation perspectives

## Citation

If you use this code in your research, please cite:

```bibtex
@article{soga2024bayesian,
  title={Constructing Bayesian Neural Network Model for Early Depression Prediction in Older Adults: Comprehensive Analysis of Humor Styles, Communication Skills, and Social Physical activity Experiences},
  author={Soga Keishi and Kawamura Takuji and Uno Akari and Kamijo Keita and Taki Yasuyuki},
  journal={In preparation},
  year={2024}
}
```

## Authors

- **Keishi Soga** - *Corresponding Author*
- Takuji Kawamura
- Akari Uno  
- Keita Kamijo
- Yasuyuki Taki

## Contact

For questions, suggestions, or collaborations, please contact:

**Keishi Soga**  
Email: keishi.soga.b4@tohoku.ac.jp

## License

This project is part of ongoing research. Please contact the authors for usage permissions and collaboration opportunities.

## Acknowledgments

This research contributes to the understanding of multifactorial approaches to mental health prediction in aging populations, combining advanced probabilistic machine learning with comprehensive psychological and behavioral assessment.

---

*Note: This is research code under active development. Results should be interpreted in the context of the specific dataset and research objectives described in the associated publication.*
