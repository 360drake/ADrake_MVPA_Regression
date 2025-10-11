# AFIS-neuro RegionalMVPA Regression
 Codebase for analyzing AFIS-neuro data as part of thesis research by Austin Drake

This directory contains Python scripts for analysis of data collected as part of the Active, Fit and Smart (AFIS) project.
"austin_utility.py" -script with callable functions: gridsearch_scoring and loo_scoring
"plot_script.py" -script for creating box plots from results: plots a score for each target (for SVR, also plots by kernel)
Model_Dataset_"gridsearch.py" -scripts for testing best parameter ranges for regression models for input MEG dataset and targets. Can include optional Metadata dataset
Model_Dataset_"analysis.py" -scripts for fitting regression model to input datasets and targets.

Intended Workflow:
1. Determine the brain imaging features to be tested. Ensure that they are in a .csv file with rows as individual subjects and columns as individual features. If no normalization of the features will be used, ensure that the feature units are consistent.
2. Check that all subjects across the brain imaging features, meta features (i.e. sex and age, if applicable) and physical activity/fitness metrics are consistent. Missing data will be handled during the analysis.
3. Run the desired input features and metrics through gridsearch code first with the model and set of hyperparameters and save the results report under Gridsearch Results. If analyzing across an important hyperparameter, i.e. testing over multiple SVM kernels, then rerun the gridsearch for each kernel.
4. From the gridsearch report(s), identify the ideal hyperparameter ranges for use in the full analysis. If the ideal range includes the minimum or maximum hyperparameter value tested, consider rerunning the gridsearch over an expanded range.
5. Input the identified hyperparamter ranges into the parameter sets in the appropriate analysis code along with the features and physical activity/fitness metrics for analysis. Save the output reports and .csv files into Results.
Notes: The results report should have all the information needed to draw your conclusions, however the various .csv files are included if it is necessary to troubleshoot the analysis.