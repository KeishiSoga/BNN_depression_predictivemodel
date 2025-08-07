####################################
# 1. Import necessary libraries
####################################
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import optax

import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import arviz as az

import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro import handlers
from numpyro.distributions import constraints
from numpyro.infer import MCMC, NUTS, Predictive


numpyro.set_platform('cpu')
numpyro.set_host_device_count(4)

# %load_ext watermark
# %watermark --iversions

####################################
# 2. Data loading
####################################
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load data
path = 'polars_df_K6.csv'
df = pd.read_csv(path)



import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Set figure size and DPI (high resolution for papers)
plt.figure(figsize=(8, 6), dpi=300)

# Data preparation
data = df['Total_K6'].dropna()

# Create histogram
n, bins, patches = plt.hist(data, bins=15, alpha=0.7, color='steelblue', 
                           edgecolor='black', linewidth=0.5, density=False)

# Calculate statistical information
mean_val = np.mean(data)
std_val = np.std(data)
median_val = np.median(data)
n_samples = len(data)

# Overlay normal distribution (optional)
x = np.linspace(data.min(), data.max(), 100)
normal_dist = stats.norm.pdf(x, mean_val, std_val) * len(data) * (bins[1] - bins[0])
plt.plot(x, normal_dist, 'r-', linewidth=2, alpha=0.8, label='Normal distribution')

# Add statistical lines
plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Mean = {mean_val:.2f}')
plt.axvline(median_val, color='orange', linestyle='--', linewidth=2, alpha=0.8, label=f'Median = {median_val:.2f}')

# Set labels and title
plt.xlabel('K6 Score', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold')
plt.title('Distribution of K6 Scores', fontsize=16, fontweight='bold', pad=20)

# Statistical information text box
#stats_text = f'N = {n_samples}\nMean = {mean_val:.2f}\nSD = {std_val:.2f}\nMedian = {median_val:.2f}'
#plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=11,
#         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Set legend
plt.legend(loc='upper right', fontsize=11)

# No grid (cleaner appearance)

# Axis settings
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.2)
plt.gca().spines['bottom'].set_linewidth(1.2)

# Tick settings
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Layout adjustment
plt.tight_layout()

# Count number of people below and above 9 points
below_9 = (df['Total_K6'] < 9).sum()
above_or_equal_9 = (df['Total_K6'] >= 9).sum()
total = below_9 + above_or_equal_9

# Calculate proportions
below_9_ratio = below_9 / total * 100
above_or_equal_9_ratio = above_or_equal_9 / total * 100

# Display results
print(f"Total participants: {total}")
print(f"Below 9 points: {below_9} people ({below_9_ratio:.2f}%)")
print(f"9 points or above: {above_or_equal_9} people ({above_or_equal_9_ratio:.2f}%)")

# Divide into below 9 points and 9 points or above

df['K6_binary'] = np.where(df['Total_K6'] < 9, 0, 1)



# Target variable
y = df['K6_binary'] 

# List of numeric and categorical features
numeric_features = [
    'Age', 'Start_age', 'Sport_years', 'work_high_MET', 'work_moderate_MET', 
    'work_METs', 'travel_METs', 'leisure_high_MET', 'leisure_moderate_MET', 
    'leisure_METs', 'Total_METs', 'Sitting',
    'Total_Affiliative_Humor', 'Total_Self_defeating_Humor', 
    'Total_Self_enhancing_Humor', 'Total_Aggressive_Humor', 
    'Total_Self_enhancing_Coping', 'Total_Cooperative_Coping', 
    'Total_Aggressive_Coping', 'Total_Self_mocking_Coping', 
    'Total_Self_Control', 'Total_Expressivity', 'Total_Decipherer', 
    'Total_Assertiveness', 'Total_Other_Acceptance', 'Total_Relationship_Regulation'
]

categorical_features = [
    'Sex','Past_ex_friends', 'Past_ex_alone', 'Present_ex_friends', 
    'Present_ex_alone'
]

# Change standardization of numeric features to Min-Max Scaler
scaler = MinMaxScaler()
X_numeric_scaled = scaler.fit_transform(df[numeric_features])



# Categorical features are already encoded as 0 and 1, so use as is
X_categorical = df[categorical_features].values

# Combine numeric and categorical features
X_processed = np.hstack((X_numeric_scaled, X_categorical))
X_full = X_processed

print(f"Full data shape: X={X_full.shape}, y={y.shape}")

####################################
# 3. Data standardization (optional recommended)
####################################
# Calculate mean and standard deviation of numeric features


# Train-test split (7:3)
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_processed,  # Numeric features after RobustScaler 
    y,            # Target after RobustScaler
    test_size=0.3,
    random_state=42
)

# Convert to JAX arrays
X_train = jnp.array(X_train_np)
X_test = jnp.array(X_test_np)
y_train = jnp.array(y_train_np)
y_test = jnp.array(y_test_np)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape : {X_test.shape},  y_test shape : {y_test.shape}")

import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC, Predictive
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def model(X, Y=None, hidden_dim=32):
    """
    3-layer (hidden) + 1 output layer MLP model (for binary classification; improved version sample)
    Args:
        X: Input data (shape: [N, D_X])
        Y: Target variable (0 or 1). Prediction mode if None.
        hidden_dim: Number of hidden layer units (expanded to default 32)
    """
    # Input dimension
    D_X = X.shape[1]
    
    # -----------------------------
    # 1. Set prior distribution variance slightly larger (0.1→0.5)
    # 2. Use LeakyReLU (activation function)
    # -----------------------------
    
    # First hidden layer
    w1 = numpyro.sample(
        "w1",
        dist.Normal(jnp.zeros((D_X, hidden_dim)), 0.5).to_event(2)
    )
    b1 = numpyro.sample(
        "b1",
        dist.Normal(jnp.zeros(hidden_dim), 0.5)
    )
    z1 = jax.nn.leaky_relu(jnp.dot(X, w1) + b1, negative_slope=0.1)
    
    # Second hidden layer
    w2 = numpyro.sample(
        "w2",
        dist.Normal(jnp.zeros((hidden_dim, hidden_dim)), 0.5).to_event(2)
    )
    b2 = numpyro.sample(
        "b2",
        dist.Normal(jnp.zeros(hidden_dim), 0.5)
    )
    z2 = jax.nn.leaky_relu(jnp.dot(z1, w2) + b2, negative_slope=0.1)
    
    # Third hidden layer
    w3 = numpyro.sample(
        "w3",
        dist.Normal(jnp.zeros((hidden_dim, hidden_dim)), 0.5).to_event(2)
    )
    b3 = numpyro.sample(
        "b3",
        dist.Normal(jnp.zeros(hidden_dim), 0.5)
    )
    z3 = jax.nn.leaky_relu(jnp.dot(z2, w3) + b3, negative_slope=0.1)
    
    # Output layer (treated as logits)
    w_out = numpyro.sample(
        "w_out",
        dist.Normal(jnp.zeros((hidden_dim, 1)), 0.5).to_event(2)
    )
    b_out = numpyro.sample(
        "b_out",
        dist.Normal(jnp.zeros(1), 0.5)
    )
    
    # Interpret z_out as logit values
    z_out = jnp.dot(z3, w_out) + b_out  # shape: [N, 1]
    
    # To save intermediate output: deterministic
    numpyro.deterministic("z_out", z_out)
    
    # Observation model (Bernoulli). Pass z_out directly to logits=...
    with numpyro.plate("data", X.shape[0]):
        numpyro.sample(
            "obs",
            dist.Bernoulli(logits=z_out.squeeze(-1)),  # shape: [N]
            obs=Y
        )


# ---- Below is an example flow for training and evaluation ----

# 1. Data preparation (example)
#    Here we assume X_train, y_train, X_test, y_test are already prepared
#    If feature scaling etc. is needed, implement accordingly
#    Convert to jnp.array for JAX
# X_train = jnp.array(X_train_np)  # example
# y_train = jnp.array(y_train_np)
# X_test  = jnp.array(X_test_np)
# y_test  = jnp.array(y_test_np)

# 2. Set NUTS settings slightly more strict and larger
nuts_kernel = NUTS(
    model,
    target_accept_prob=0.9,  # Increase from 0.8 → 0.9 (sampling stabilization)
    max_tree_depth=12        # Increase tree depth to 12 (default is 10)
)

mcmc = MCMC(
    nuts_kernel,
    num_warmup=1500,   # Warmup to 1500
    num_samples=3000,  # Number of samples to 3000 
    num_chains=4,
    chain_method='parallel',
    progress_bar=True
)

# 3. Execute MCMC
rng_key = random.PRNGKey(0)
mcmc.run(rng_key, X=X_train, Y=y_train, hidden_dim=32)
mcmc.print_summary()

# 4. Prediction
posterior_samples = mcmc.get_samples()

# Extract z_out (logits) with Predictive
predictive = Predictive(model, posterior_samples, return_sites=["z_out", "obs"])
rng_key_pred = random.PRNGKey(1)
samples = predictive(rng_key_pred, X=X_test, Y=None, hidden_dim=32)

# Bernoulli(0/1) samples
y_pred_samples = samples["obs"]  # shape: (num_samples, N_test)

# Convert from logit (z_out) samples to probabilities if needed
z_out_samples = samples["z_out"]  # shape: (num_samples, N_test, 1)
p_samples = jax.nn.sigmoid(z_out_samples.squeeze(-1))  # shape: (num_samples, N_test)

# 5. Lower threshold below 0.5 to increase Recall (example)
threshold = 0.3
p_mean = jnp.mean(p_samples, axis=0)  # mean over MCMC samples
y_pred_label = (p_mean >= threshold).astype(int)

# 6. Evaluation
y_true = np.array(y_test)         # sklearn.metrics prefers numpy array
y_pred = np.array(y_pred_label)   # same as above

acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec  = recall_score(y_true, y_pred)
f1   = f1_score(y_true, y_pred)

print("\n=== Evaluation metrics for binary classification on test data (threshold=", threshold, ")===")
print(f"Accuracy : {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall   : {rec:.3f}")
print(f"F1-score : {f1:.3f}")

# 7. Visualization of confusion matrix (example)
cm = confusion_matrix(y_true, y_pred)
plt.imshow(cm)  # Default colors. Principle is not to set details
plt.title("Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("True label")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.show()


import arviz as az

# Convert to ArviZ format
idata = az.from_numpyro(mcmc)

# Display summary
print(az.summary(idata, var_names=["b1", "b_out", "w1", "w2", "w3", "w_out"]))

import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from numpyro.infer import Predictive
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# 1. Use posterior samples and test data to sample predictions with Predictive
predictive = Predictive(model, posterior_samples)
rng_key_pred = PRNGKey(0)
samples = predictive(rng_key_pred, X=X_test, Y=None, hidden_dim=32)

# 2. samples["obs"] contains 0/1 with shape: (num_samples, N_test)
y_pred_samples = samples["obs"]

# 3. Calculate prediction probabilities (occurrence rate of 1 for each sample)
#    Since y_pred_samples is 0 or 1 sampling, averaging in axis=0 direction gives "probability of 1"
y_pred_prob = jnp.mean(y_pred_samples, axis=0)  # shape: (N_test,)

# 4. Determine final labels with threshold 0.5
y_pred_label = (y_pred_prob >= 0.5).astype(int)  # shape: (N_test,)

# 5. Calculate evaluation metrics (assuming y_test is 0/1 numpy array)
acc  = accuracy_score(y_test, y_pred_label)
prec = precision_score(y_test, y_pred_label)
rec  = recall_score(y_test, y_pred_label)
f1   = f1_score(y_test, y_pred_label)

print("=== Evaluation metrics for binary classification on test data ===")
print(f"Accuracy : {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall   : {rec:.3f}")
print(f"F1-score : {f1:.3f}")

# 6. Visualize confusion matrix
cm = confusion_matrix(y_test, y_pred_label)
plt.imshow(cm, cmap="Blues")  # Can be omitted if "no color specification" is the principle
plt.title("Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("True label")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
plt.show()

# 7. Histogram to check probability distribution (example)
plt.hist(y_pred_prob, bins=20, alpha=0.6)
plt.title("Predicted Probability Distribution (for Y=1)")
plt.xlabel("Predicted Probability (y=1)")
plt.ylabel("Count")
plt.grid(True)
plt.show()

import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from numpyro.infer import Predictive
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# 1. Use posterior samples and test data to sample predictions with Predictive
predictive = Predictive(model, posterior_samples)
rng_key_pred = PRNGKey(0)
samples = predictive(rng_key_pred, X=X_test, Y=None, hidden_dim=32)

# 2. samples["obs"] contains 0/1 with shape: (num_samples, N_test)
y_pred_samples = samples["obs"]

# 3. Calculate prediction probabilities (occurrence rate of 1 for each sample)
#    Since y_pred_samples is 0 or 1 sampling, averaging in axis=0 direction gives "probability of 1"
y_pred_prob = jnp.mean(y_pred_samples, axis=0)  # shape: (N_test,)

# 4. Determine final labels with threshold 0.3
y_pred_label = (y_pred_prob >= 0.3).astype(int)  # shape: (N_test,)

# 5. Calculate evaluation metrics (assuming y_test is 0/1 numpy array)
acc  = accuracy_score(y_test, y_pred_label)
prec = precision_score(y_test, y_pred_label)
rec  = recall_score(y_test, y_pred_label)
f1   = f1_score(y_test, y_pred_label)

print("=== Evaluation metrics for binary classification on test data ===")
print(f"Accuracy : {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall   : {rec:.3f}")
print(f"F1-score : {f1:.3f}")

# 6. Visualize confusion matrix
cm = confusion_matrix(y_test, y_pred_label)
plt.imshow(cm, cmap="Blues")  # Can be omitted if "no color specification" is the principle
plt.title("Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("True label")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
plt.show()

# 7. Histogram to check probability distribution (example)
plt.hist(y_pred_prob, bins=20, alpha=0.6)
plt.title("Predicted Probability Distribution (for Y=1)")
plt.xlabel("Predicted Probability (y=1)")
plt.ylabel("Count")
plt.grid(True)
plt.show()

thresholds = np.linspace(0.15, 0.3, 50)
f1_scores = []

for thresh in thresholds:
    y_pred_label_thresh = (y_pred_prob >= thresh).astype(int)
    f1 = f1_score(y_test, y_pred_label_thresh)
    f1_scores.append(f1)

# Plot
plt.plot(thresholds, f1_scores, marker='o')
plt.title("F1-score vs Threshold")
plt.xlabel("Threshold")
plt.ylabel("F1-score")
plt.grid(True)
plt.show()

# Optimal threshold and F1-score at that time
best_idx = np.argmax(f1_scores)
best_thresh = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"Optimal threshold: {best_thresh:.3f}")
print(f"F1-score at that time: {best_f1:.3f}")


 #1. First optimize the threshold
thresholds = np.linspace(0.15, 0.3, 50)
f1_scores = []

for thresh in thresholds:
    y_pred_label_thresh = (y_pred_prob >= thresh).astype(int)
    f1 = f1_score(y_test, y_pred_label_thresh)
    f1_scores.append(f1)

# Identify optimal threshold
best_idx = np.argmax(f1_scores)
best_thresh = thresholds[best_idx]
print(f"Optimal threshold: {best_thresh:.3f}")

# 2. Perform prediction with optimal threshold
y_pred_label = (y_pred_prob >= best_thresh).astype(int)

# 3. Calculate evaluation metrics
acc  = accuracy_score(y_test, y_pred_label)
prec = precision_score(y_test, y_pred_label)
rec  = recall_score(y_test, y_pred_label)
f1   = f1_score(y_test, y_pred_label)

print("\n=== Evaluation metrics with optimal threshold ===")
print(f"Accuracy : {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall   : {rec:.3f}")
print(f"F1-score : {f1:.3f}")

# 4. Visualization of confusion matrix
cm = confusion_matrix(y_test, y_pred_label)
plt.imshow(cm, cmap="Blues")
plt.title(f"Confusion Matrix (threshold={best_thresh:.3f})")
plt.xlabel("Predicted label")
plt.ylabel("True label")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
plt.show()


import jax.numpy as jnp

# y_pred_samples = samples["obs"]  # shape: (num_samples, N_test)

# Average of 0/1 samples (= probability estimate of 1 occurrence)
y_pred_mean = jnp.mean(y_pred_samples, axis=0)  # shape: (N_test, )

# Calculate 95% credible interval (2.5%, 97.5%) with 0/1 samples
y_pred_lower = jnp.percentile(y_pred_samples, 2.5, axis=0)   # shape: (N_test, )
y_pred_upper = jnp.percentile(y_pred_samples, 97.5, axis=0)  # shape: (N_test, )

# y_pred_lower, y_pred_upper tend to be mostly 0 or 1


# 1. Set up to obtain z_out samples with Predictive
predictive = Predictive(
    model, 
    posterior_samples, 
    return_sites=["z_out"]  # Can also add "obs" etc.
)
rng_key_pred = random.PRNGKey(0)
samples = predictive(rng_key_pred, X=X_test, Y=None, hidden_dim=32)

z_out_samples = samples["z_out"]  # shape: (num_samples, N_test, 1) etc.

# 2. Sigmoid transform to probability p
p_samples = jax.nn.sigmoid(z_out_samples.squeeze(-1))  
# shape: (num_samples, N_test)


import matplotlib.pyplot as plt

# Mean and 2.5%, 97.5% percentiles of probability distribution
p_mean = jnp.mean(p_samples, axis=0)                # shape: (N_test,)
p_lower = jnp.percentile(p_samples, 2.5, axis=0)    # shape: (N_test,)
p_upper = jnp.percentile(p_samples, 97.5, axis=0)   # shape: (N_test,)

x_vals = jnp.arange(len(y_test))

plt.figure(figsize=(10,5))
plt.errorbar(
    x_vals,
    p_mean,
    yerr=[p_mean - p_lower, p_upper - p_mean],
    fmt='o',
    ecolor='gray',
    capsize=3,
    label='Predicted p (mean ± 95% CI)'
)
plt.scatter(x_vals, y_test, color='red', alpha=0.6, label='True label (0 or 1)')
plt.xlabel('Test sample index')
plt.ylabel('Predicted Probability of Y=1')
plt.title('Predictive distribution (Bernoulli) with 95% Credible Interval')
plt.legend()
plt.show()


# Assumes continuous "prediction probability" (p_samples) distribution exists
# p_mean, p_lower, p_upper = mean and percentiles in axis=0 direction for posterior samples

x_vals = jnp.arange(len(y_test))

plt.figure(figsize=(10, 5))
plt.plot(x_vals, p_mean, label='Predicted probability of Y=1')
plt.fill_between(x_vals, p_lower, p_upper, color='gray', alpha=0.3, label='95% CI')
plt.scatter(x_vals, y_test, color='red', alpha=0.6, label='True label (0 or 1)')
plt.xlabel('Test sample index')
plt.ylabel('Probability')
plt.title('Predictive Probability with 95% Credible Interval')
plt.legend()
plt.show()

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor


az.to_netcdf(idata, filename='idata_EX_K6_5_2_bina_9_5_19.nc')

# ❶ Hold posterior samples in closure
predictive = Predictive(model, posterior_samples)


def bnn_predict(x_np: np.ndarray) -> np.ndarray:
    """
    Prediction function called from SHAP
    x_np : NumPy array with shape (n_samples, n_features)
    Return value: Probabilities (0–1) with shape (n_samples,)
    """
    # NumPy → JAX
    x_jax = jnp.asarray(x_np)

    # Prediction (pattern that folds parameter uncertainty with average)
    rng_key = PRNGKey(0)                      # Change random seed each time if needed
    samples  = predictive(rng_key, X=x_jax, Y=None, hidden_dim=32)
    probs    = jnp.mean(samples["obs"], axis=0)  # shape (n_samples,)

    return np.asarray(probs)                  # SHAP expects NumPy


import shap
# Concatenate numeric and categorical lists
feature_names = numeric_features + categorical_features
print(len(feature_names), "features")        # ← Confirm it matches X_test_np.shape[1]

# Use 500 people as background ← Good balance of computational reliability and speed
masker = shap.maskers.Independent(X_train_np, max_samples=500)
explainer = shap.Explainer(bnn_predict, masker, output_names=["P(Y=1)"])

# Calculate SHAP values for 500 people in test data
shap_values = explainer(X_test_np[:500], max_evals=100)

# Visualization (summary plot)
shap.summary_plot(shap_values, X_test_np[:500], feature_names=feature_names)


X_slice = X_test_np[:500]          # ← Same 500 rows passed to explainer

shap.summary_plot(
    shap_values,                   # Explanation main body
    X_slice,                       # Align number of rows!
    feature_names=feature_names,
    plot_type="violin",             
    max_display=len(feature_names)  
)
