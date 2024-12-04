# Project Overview

This project aims to:

Analyze air quality data to identify patterns and features that influence air pollution levels.
Implement and compare the performance of two machine learning models:
Decision Tree Classifier
K-Nearest Neighbors (KNN)
Address class imbalance and skewed feature distributions using appropriate preprocessing techniques.

# Dataset Description
The dataset used in this project is the Air Quality UCI dataset, which contains hourly measurements of various air pollutants and meteorological data. Key features include:

CO(GT): True hourly average CO concentration (mg/m^3).
NOx(GT): True hourly average NOx concentration (ppb).
NO2(GT): True hourly average NO2 concentration (microg/m^3).
T: Temperature (°C).
RH: Relative humidity (%).
AH: Absolute humidity (g/m^3).
Key Statistics:
Number of samples: 9,358
Missing Values: Replaced using mean/mode imputation.
Target Variable: Air_Quality_Level (Good, Moderate, Poor).

# Preprocessing Steps
To ensure reliable and accurate predictions, the following preprocessing steps were applied:

Handling Missing Values:
Missing values in pollutant features (e.g., CO, NOx) were replaced using mean or mode imputation.
Addressing Class Imbalance:
The dataset was balanced using Synthetic Minority Oversampling Technique (SMOTE).
Transforming Skewed Features:
Log transformation was applied to handle skewness in pollutant data.
Feature Scaling:
Data was standardized using StandardScaler for both Decision Tree and KNN to ensure features are on the same scale.
Dimensionality Reduction (KNN only):
PCA was applied to reduce feature dimensions while retaining 95% of variance.

# Models Used
1. Decision Tree Classifier
Parameters:
max_depth=9
min_samples_split=10
min_samples_leaf=5
class_weight='balanced'
Evaluation Metrics:
Accuracy: 98%
Cross-Validation Accuracy: 94%
Confusion Matrix: Displays predictions for Good, Moderate, and Poor air quality levels.

# 2. K-Nearest Neighbors (KNN)
Parameters:
Distance metric: Manhattan
Optimal number of neighbors (k): 5
Evaluation Metrics:
Accuracy: 93%
Cross-Validation Accuracy: 88%
Confusion Matrix: Shows high recall and precision for all classes after preprocessing.


