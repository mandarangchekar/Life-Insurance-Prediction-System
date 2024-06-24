# Life-Insurance-Prediction-System

This project aims to compare various machine learning models to predict a binary target based on a set of features. We've tested models such as Random Forest, Gradient Boosting, XGBoost, Logistic Regression, and ensemble methods like Voting and Stacking Classifiers to identify the best performer based on metrics like ROC, accuracy, log loss, F-Score, precision, and recall.

## Project Overview

### Objective

The goal is to evaluate different models on their ability to predict a binary target from provided dataset features effectively. The comparison will help in selecting the most suitable model based on various performance metrics.

### Data Description

The dataset used in this project contains numerous features that have been anonymized and represent different attributes collected from the subjects. The target variable is binary.

### Models Evaluated

- Random Forest
- Gradient Boosting
- XG Boost
- Logistic Regression
- Voting Classifier
- Stacked Model

## Analysis

### Data Preparation

Data preparation involved handling missing values, encoding categorical variables, feature scaling, and splitting the dataset into training and testing sets.

### Model Building

Each model was implemented with specific parameters, optimized using GridSearchCV when applicable. The following steps were taken for each model:
- Parameter tuning
- Model fitting
- Predictions and evaluations on the test set

### Model Evaluation Metrics

Models were evaluated based on the following metrics:
- ROC AUC
- Accuracy
- Log Loss
- F-Score
- Precision
- Recall

## Results

The models' performance was summarized in the table below:

| Model Name            | Train ROC | Test ROC | Train Accuracy | Test Accuracy | Train Log Loss | Test Log Loss | F-Score | Precision | Recall |
|-----------------------|-----------|----------|----------------|---------------|----------------|---------------|---------|-----------|--------|
| Random Forest         | 0.757     | 0.751    | 0.808          | 0.806         | 0.426          | 0.428         | 0.666   | 0.755     | 0.595  |
| Gradient Boosting     | 0.840     | 0.816    | 0.854          | 0.834         | 0.318          | 0.352         | 0.749   | 0.737     | 0.763  |
| XG Boost              | 0.810     | 0.806    | 0.828          | 0.826         | 0.364          | 0.367         | 0.737   | 0.725     | 0.748  |
| Logistic Regression   | 0.695     | 0.698    | 0.746          | 0.751         | 0.499          | 0.495         | 0.587   | 0.635     | 0.546  |
| Voting Classifier     | 0.806     | 0.797    | 0.835          | 0.829         | 0.381          | 0.389         | 0.729   | 0.753     | 0.706  |
| Stacked Model         | 0.814     | 0.794    | 0.837          | 0.822         | 0.353          | 0.412         | 0.722   | 0.731     | 0.713  |

## Conclusion

Based on our evaluation, Gradient Boosting has shown the best overall performance, especially in terms of ROC and accuracy. It also maintains a low log loss compared to other models. Future work could explore further parameter tuning or using different ensemble methods to improve model accuracy.

## Usage

Details on how to run the models and recreate the analysis can be found in the Jupyter notebooks included in the repository.

## Requirements

- Python 3.x
- Pandas
- Numpy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn

