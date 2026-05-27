# Predicting Daily Household Energy Consumption Using Kernel Methods and Gaussian Process Regression

A study comparing Kernel Ridge Regression (KRR) and Gaussian Process Regression (GPR) for forecasting daily household energy consumption, evaluating both univariate (lag-based) and multivariate (covariate-augmented) formulations against naive baselines.

## Overview

Accurate forecasting of residential energy consumption is essential for demand planning, grid stability, and energy-efficient resource allocation. Linear models struggle to capture the nonlinear interactions and seasonal effects in electrical consumption data, so this project applies kernel-based learning methods that operate in reproducing kernel Hilbert spaces (RKHS) to model these dynamics on the UCI Household Electric Power Consumption dataset.

The goals of the project are:

1. Quantify the benefits of multivariate modeling over univariate modeling.
2. Compare the performance of RBF, Polynomial, Periodic, Matérn, and composite kernels.
3. Analyze GPR hyperparameters and posterior variance for interpretability.

## Dataset

The dataset used is the **UCI Household Electric Power Consumption** dataset, which contains minute-level measurements of electrical consumption in a single household from December 2006 to November 2010. The raw data includes global active and reactive power, voltage, intensity, and three sub-metering variables.

**Source:** UCI Machine Learning Repository — *Individual Household Electric Power Consumption*, 2012.
Available at: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

> **Note:** The dataset file (`household_power_consumption.txt`) is **not included in this repository** due to size. Please download it directly from the UCI link above and place it in the project root before running the notebook.

### Preprocessing

- Missing values (encoded as `"?"`) are imputed via forward and backward filling.
- All measurements are converted to floats and combined under a unified timestamp index.
- Minute-level data is resampled to **daily averages** to reduce noise and align with long-term consumption trends.
- Features are standardized to zero mean and unit variance using statistics estimated from the training set only.

### Feature Sets

**Univariate features** (lagged daily consumption):

$$X_t = \{y_{t-1},\ y_{t-7},\ y_{t-30}\}$$

**Multivariate features** (6-dimensional lagged covariates):

$$Z_t = [\text{ReactivePower}_{t-1},\ \text{Voltage}_{t-1},\ \text{Intensity}_{t-1},\ \text{Sub1}_{t-1},\ \text{Sub2}_{t-1},\ \text{Sub3}_{t-1}] \in \mathbb{R}^{6}$$

## Methodology

### Kernel Ridge Regression (KRR)

KRR combines ridge regression with kernel methods in an RKHS. Kernels evaluated:

- **RBF kernel:** `k(x, x') = exp(-||x - x'||² / 2ℓ²)`
- **Polynomial kernel:** `k(x, x') = (xᵀx' + c)ᵈ`

Hyperparameters (λ, γ, degree, bias) are tuned via grid search using `TimeSeriesSplit` cross-validation with 5 folds.

### Gaussian Process Regression (GPR)

GPR provides a Bayesian treatment with full posterior predictive distributions (mean and variance). Kernels evaluated:

- **RBF**
- **Periodic** (ExpSineSquared)
- **RBF + Periodic** (additive composite)
- **Matérn** (ν = 1.5)

GPR hyperparameters (length-scale ℓ, signal variance, noise variance σₙ², periodicity p) are optimized by maximizing the log-marginal likelihood.

### Baselines

- **Naive:** `ŷ_t = y_{t-1}`
- **MA(7):** constant forecast equal to the final 7-day rolling mean of the training period

## Results

Models evaluated on the 2010 test set, measured by RMSE and MAE:

| Model                       | RMSE   | MAE    |
| --------------------------- | ------ | ------ |
| **MULTI — KRR Poly**        | **0.2508** | **0.1856** |
| MULTI — KRR RBF             | 0.2521 | 0.1902 |
| MULTI — GPR RBF             | 0.2524 | 0.1860 |
| MULTI — GPR Matérn          | 0.2526 | 0.1874 |
| UNI — GPR RBF               | 0.2539 | 0.1902 |
| UNI — GPR RBF+Periodic      | 0.2550 | 0.1910 |
| UNI — KRR Poly              | 0.2561 | 0.1926 |
| UNI — KRR RBF               | 0.2566 | 0.1954 |
| UNI — GPR Periodic          | 0.2588 | 0.1970 |
| UNI — Naive                 | 0.3032 | 0.2200 |
| UNI — MA(7)                 | 0.5710 | 0.4907 |

### Key Findings

- **Multivariate models outperform univariate models** across all kernel choices. Reactive power, voltage, intensity, and sub-metering variables contain predictive information not captured by lagged consumption alone.
- **Multivariate KRR with a Polynomial kernel** achieves the best overall performance (RMSE = 0.2508, MAE = 0.1856), indicating that nonlinear feature interactions across covariates matter.
- **RBF-based GPR models perform almost identically to KRR with RBF kernels**, consistent with their shared quadratic RKHS penalty — but GPR additionally provides uncertainty estimates through posterior variance.
- **Purely Periodic kernels underperform** in the univariate setting because daily averaging smooths out sharp seasonal effects and introduces slow non-stationary trends that are not purely periodic.

## Repository Contents

```
.
├── README.md                          # This file
├── model.ipynb                        # Main notebook (preprocessing, training, evaluation)
├── report.pdf                         # Full academic report
├── Project_Presentation_Animesh.pptx  # Project presentation slides
├── fig_daily_series.png               # Daily average global active power over time
├── fig_rmse_bar.png                   # RMSE comparison across all models
├── fig_uni_krrpoly.png                # Univariate KRR (Polynomial) — test predictions
└── fig_multi_krrpoly.png              # Multivariate KRR (Polynomial) — test predictions
```

## Setup

### Requirements

- Python 3.9+
- NumPy
- pandas
- scikit-learn
- matplotlib
- Jupyter

Install dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib jupyter
```

### Running the Project

1. Download `household_power_consumption.txt` from the [UCI repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) and place it in the project root.
2. Launch the notebook:
   ```bash
   jupyter notebook model.ipynb
   ```
3. Run all cells to reproduce preprocessing, training, evaluation, and figures.

## Future Work

- Multi-output GPR models for jointly forecasting multiple electrical variables.
- Deep kernel learning to combine neural feature extractors with kernel methods.
- Locally periodic kernels to better capture slowly evolving seasonal patterns.
- Incorporating exogenous variables such as weather data or electricity pricing.

## References

1. UCI Machine Learning Repository, "Individual household electric power consumption," 2012.
2. C. E. Rasmussen and C. K. I. Williams, *Gaussian Processes for Machine Learning*. MIT Press, 2006.
3. J. Shawe-Taylor and N. Cristianini, *Kernel Methods for Pattern Analysis*. Cambridge University Press, 2004.

## Authors

Animesh Kashid

Northeastern University
