Attention-LSTM Time Series Forecasting

📌 Project Overview

This project implements a Deep Learning Time Series Forecasting pipeline designed to predict macroeconomic indicators (specifically Real GDP) using a Long Short-Term Memory (LSTM) network augmented with a Self-Attention Mechanism.

The model is built using PyTorch and is benchmarked against a Standard LSTM (without attention) and Prophet (optional) to demonstrate the efficacy of attention mechanisms in capturing long-term dependencies and seasonal patterns in multivariate time series data.

🚀 Key Features

Self-Attention Mechanism: A custom PyTorch module that weighs the importance of different time steps in the input sequence, allowing the model to focus on relevant historical context.

Multivariate Forecasting: Uses multiple macroeconomic variables (Consumption, Investment, CPI, Unemployment) to predict GDP.

Robust Backtesting: Implements Rolling-Window Cross-Validation (3 sequential holdout folds) to ensure performance stability.

Hyperparameter Optimization: Integrated Optuna support for Bayesian optimization of learning rate, hidden dimensions, dropout, and layers.

Production-Ready Pipeline: Single-file architecture including data loading, scaling, feature engineering, training, evaluation, and reporting.

📂 Dataset

The project uses the statsmodels.macrodata dataset, which contains US macroeconomic time series data since 1959.

Target: realgdp (Real Gross Domestic Product)

Features: realcons (Consumption), realinv (Investment), cpi (Inflation), unemp (Unemployment)

Derived Features: Lags (1-4 quarters), Rolling Means, and Rolling Standard Deviations.

🛠️ Installation & Requirements

Ensure you have Python 3.8+ installed. Install the dependencies using pip:

pip install torch pandas numpy scikit-learn statsmodels matplotlib optuna tqdm


Optional (for baseline comparison):

pip install prophet


🏃‍♂️ Usage

The entire pipeline is contained within a single Python script. You can run it directly from the terminal or an IDE.

1. Standard Run

Runs the training and backtesting with default hyperparameters.

# Inside your script or main block
run_pipeline(save_dir='./results', seq_len=8, tune=False)


2. Run with Hyperparameter Tuning

Enables Optuna to search for the best model configuration before backtesting.

# Inside your script or main block
run_pipeline(save_dir='./results_tuned', seq_len=8, tune=True)


📊 Methodology

1. Data Preprocessing

Windowing: The data is transformed into sequences of length seq_len (default 8 quarters).

Scaling: StandardScaler is fitted on the training split of each fold to prevent data leakage.

Inverse Transformation: Predictions are inverse-transformed to their original scale (Real GDP values) for interpretable error metrics.

2. Model Architecture

Input Layer: Accepts sequence shape (Batch, Seq_Len, Features).

LSTM Layer: Processes temporal dependencies.

Attention Layer: Computes a weighted sum of LSTM hidden states across all time steps.

$Score = H \cdot H^T$

$Weights = Softmax(Score)$

$Context = Weights \cdot H$

Output Layer: Fully connected layer mapping the context vector to the target value.

3. Evaluation Metrics

The model is evaluated using the following metrics across all backtesting folds:

MAE (Mean Absolute Error): Average magnitude of errors.

RMSE (Root Mean Squared Error): Penalizes larger errors.

CFE (Cumulative Forecast Error): Measures bias (over/under-prediction).

📈 Visualization

The pipeline automatically generates comparison plots for each fold in the results directory:

Black Line: True Historical GDP.

Red Line: Attention-LSTM Prediction.

Blue Dashed Line: Standard LSTM Prediction.

📁 Output Structure

After running the pipeline, a results folder is created:

results/
├── fold0_preds.png      # Visualization of the first backtest fold
├── results.json         # detailed metrics and configuration
├── REPORT.md            # Auto-generated text summary of the run
└── SUMMARY.md           # Conclusion on Attention mechanism impact
