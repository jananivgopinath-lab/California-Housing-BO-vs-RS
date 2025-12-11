# California-Housing-BO-vs-RS
Bayesian Optimization vs Random Search
Hyperparameter Tuning on XGBoost – California Housing Dataset

This project demonstrates how to compare Bayesian Optimization (using gp_minimize) with Random Search for tuning an XGBoost regression model.
The California Housing dataset is used because it is clean, purely numerical, and well-suited for evaluating optimization methods.

Project Goals

Build an XGBoost regression model

Tune hyperparameters using
* Bayesian Optimization (gp_minimize)
* Random Search (baseline)

Compare

* RMSE performance
* Convergence behavior
* Parameter efficiency
* Visualize optimization progress

Dataset

California Housing Dataset (Scikit-Learn)
8 continuous numerical features
No missing values
Predicts median house value
Good dataset for testing optimization algorithms

Preprocessing

* Features scaled using StandardScaler
3-Fold Cross-Validation (KFold(n_splits=3))
RMSE used as evaluation metric

Since the dataset has no missing values, imputation was not required.

Hyperparameters Tuned

Parameter     	Range
n_estimators	   50 – 400
max_depth     	 2 – 10
learning_rate 	 0.01 – 0.3
subsample	       0.5 – 1.0
colsample_bytree 0.5 – 1.0

Both Bayesian Optimization and Random Search use equal budget (20 iterations).

Methods Compared
1. Bayesian Optimization (gp_minimize)
Uses Gaussian Process–based surrogate modeling to intelligently choose the next best hyperparameters.

Advantages
* Learns from previous trials
* Explores → then focuses
* Fewer wasted evaluations

2. Random Search
Samples hyperparameters uniformly and independently.

Advantages
* Simple
* Faster
* No modeling overhead

Result Summary
Bayesian Optimization typically outperformed Random Search because it uses information from earlier trials to guide the search.

Convergence Plot
Both optimization curves are plotted to show how RMSE changed across iterations:

BO shows smoother, improving trend

Random Search jumps randomly with high variance

This plot clearly highlights how informed search outperforms uninformed sampling.
