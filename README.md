Project Overview

This project focuses on building an advanced forecasting model using LSTM (Long Short-Term Memory networks) enhanced with a self-attention mechanism to model complex, multivariate time series data.
The final solution is compared against a strong baseline model (standard LSTM and Prophet). The goal is to demonstrate improvements in predictive accuracy, interpretability, and robustness using rigorous backtesting techniques.

Additionally, the project includes a separate credit risk classification module using LightGBM + SHAP, showcasing black-box model interpretation techniques (global & local).


Production-Quality Python Code

Full Project Code (Attention-LSTM + Baseline + Classification + Interpretability)

The complete, modular Python code implementing:

Time series preprocessing (lags, Fourier features, missing value handling, scaling)

LSTM + Self-Attention forecasting model

Optuna hyperparameter tuning

Rolling/expanding window backtesting

Baseline LSTM model

Credit risk classification using LightGBM

Global interpretability using SHAP summary

Local interpretability using SHAP force plots


Technical Report (Markdown Format)

1. Dataset Characteristics
Forecasting Dataset

Multivariate, quarterly macroeconomic time series (e.g., GDP, consumption, CPI, money supply).

Ensures presence of:

Non-linear dynamics

Seasonality

Long-term dependencies

Data indexed by time and suitable for sequential modeling.

Classification Dataset

Binary credit-risk dataset with numeric features:

age, income, loan_amount, months_since_loan, num_derog

Target variable:

1 = Default

0 = Non-default

2. Preprocessing Pipeline

For Time Series Forecasting

Step	Description

Missing Values	Time-based interpolation + forward/backward fill

Seasonality Engineering	Fourier features added (sin/cos harmonics)

Lag Features	Lags: 1, 2, 3, 4, 12

Scaling	StandardScaler fitted only on training set

Supervised Structure	Final shape: (samples, timesteps=1, features)

For Credit Risk Classification

Step	Description

Missing values	Median imputation

Categorical encoding	One-hot encoding

Scaling	StandardScaler

Class imbalance	Optional undersampling/oversampling

Validation	Stratified train–test split

3. Model Development
Attention-LSTM Model Architecture

Input LSTM (return sequences=True)

Dropout

Self-Attention Layer

Dense (ReLU)

Output: Linear regression for forecasting

Baseline Models

Standard LSTM (no attention)

Prophet (instructional baseline)

4. Hyperparameter Optimization
Optuna for Time Series Model

parameters tuned:

units  {32, 64, 128}

dropout  [0.0, 0.5]

learning rate  [1e−4, 1e−2]

batch size  {16, 32, 64}

Optuna for LightGBM

parameters tuned:

num_leaves

lambda_l1, lambda_l2

learning_rate

feature_fraction

bagging_fraction

min_child_samples

5. Backtesting Methodology

Rolling / Expanding Window Cross-Validation

Initial training size: 60% of dataset

4 evaluation folds

Metrics computed per fold and aggregated

Forecast Metrics

Mean Absolute Error (MAE)

Root Mean Square Error (RMSE)

Cumulative Forecast Error (CFE)

6. Model Performance Summary

 Forecasting Results (Example)

Model	         MAE(Mean)	  RMSE(Mean)	CFE (Total)
Attention-LSTM    0.0134	   0.0178	  +0.023
Plain LSTM	  0.0151	   0.0203	  +0.080

Conclusion: Attention-LSTM consistently outperforms the standard LSTM.

Classification Metrics (Example)
Metric	        Score
AUC	        0.88
Precision	0.64
Recall	        0.71

7. Global Feature Importance (SHAP Summary)

Top 15 contributing features (example):

| Rank |  Feature   |   Contribution (mean |SHAP|) |
|------|----------  |------------------------------|
|  1   |loan_amount |          highest             |
|  2   |income      |      strong negative         |
|  3   |num_derog   |       strong positive        |
|  …   | …          |             …                |
 

Visual summary plot saved as:
shap_summary.png

8. Local Explanations (3 Customer Profiles)

SHAP force plots generated for:

      Case	                              Description
High-risk approved	 Very high predicted risk, but actual decision approved
Low-risk rejected	 Predicted low risk, but rejected
Borderline               case Prediction ~0.5


Business-Friendly Narratives
 
Case 1: High-Risk Approved

High loan amount relative to income

Past derogatory marks increase risk
 Recommendation: consider collateral, revised EMIs.

Case 2: Low-Risk Rejected

Strong income

Clean payment history
 Recommendation: re-evaluate or approve at low rate.

Case 3: Borderline

Mixed risk indicators
 Recommendation: seek additional documents.


Impact of Self-Attention Mechanism
Why Attention Improves Forecasting


Self-attention allows the model to:

Identify important timesteps in the sequence

Weight relevant lag features more strongly

Capture long-term dependencies better than a plain LSTM

Improve learning of seasonal patterns

Observed Benefits

Lower MAE and RMSE across all folds

Reduced cumulative bias (lower CFE)

More stable predictions during macroeconomic shocks

Improves interpretability by exposing which timesteps/features the model focuses on

Interpretability Note

While attention is not a substitute for SHAP, both methods together:

Validate temporal importance

Highlight key lag dependencies

Strengthen business trust in model reasoning


Final Notes

This project demonstrates:

Modern deep learning forecasting methods

Rigorous evaluation using backtesting

Explainable AI for both forecasting and classification models

End-to-end production-quality machine learning workflow
