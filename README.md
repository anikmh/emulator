# Overview
This project currently aims to generate emulators to perform tasks such as solving the TOV equations, obtaining the mass-radius curves (inverse TOV), and Bayesian inference. In high dimensions, these tasks can be expensive in scientific computations. Thus, the goal of this project is to mitigate the curse of high dimensionality by using powerful machine-learning techniques such as neural networks, decision tree regressors, and gaussian processes.

# Structure
The directories in this repository are organized as follows:
 - `checkpoints/` contains saved profiles during model fittings to continue later from where left off
 - `include/` stores necessary headers and module imports for bookkeeping
 - `data/` processes raw data and temporarily stores the processed data files, which may be empty to save memory
 - `search/` performs grid searches over the hyperparameters to tune models and saves the best model(s)
 - `train/` fits model using results from grid search or user-specified models and saves the trained model
 - `evaluate/` loads the saved models and evaluates new points

# Updates
This section will contain continuous updates on version changes, new features, and bug fixes, as the project develops.