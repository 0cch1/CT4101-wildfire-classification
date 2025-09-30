# CT4101 Assignment 1 - Classification using Scikit-learn

## Overview
This project is part of **Machine Learning CT4101**.
The goal is to implement classification models using **Scikit-learn**, train them on the provided wildfire datasets, and evaluate their performance under different hyperparameter settings.

This repository contains my implementation for **Algorithm 1 - Logistic Regression**.
A second algorithm will be implemented separately in the same structure.

---

## Dataset
Two csv files:
- `wildfire_training.csv`
- `wildfire_test.csv`

### Columns
fire (target: yes/no), year, temp, humidity, rainfall, drought_code, buildup_index, day, month, wind_speed

The task is to predict whether a wildfire(`fire`) occurs based on the other attributes.

---

## How to run

1. Install dependencies (better in a virtual environment):
  pip install -r requirements.txt
or manually:
  pip install pandas numpy scikit-learn matplotlib

2. Run the Logistic Regression script:
  python ct1401.assignment.py