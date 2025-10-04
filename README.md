# CT4101 Assignment 1 - Classification using Scikit-learn

## Overview
Implementation of two classification algorithms on wildfire prediction dataset:
- **Logistic Regression** with hyperparameter tuning
- **Random Forest** with 2 hyperparameters only (assignment requirement)

---

## Files

- `ct4101_assignment.py` - Helper functions
- `logisticRegression.ipynb` - Complete Logistic Regression analysis
- `randomForest.ipynb` - Complete Random Forest analysis
- `wildfires_training.csv` - Training data (77 samples)
- `wildfires_test.csv` - Test data (50 samples)

---

## Dataset
Features: year, temp, humidity, rainfall, drought_code, buildup_index, day, month, wind_speed
Target: fire (yes/no)

---

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run Jupyter notebooks:
```bash
jupyter notebook
# Open and run logisticRegression.ipynb and randomForest.ipynb
```

---

## Results

### Logistic Regression
- Best: penalty='none', C=1.0
- Test Accuracy: 90%

### Random Forest
- Best: max_depth=4, min_samples_leaf=4
- Test Accuracy: 86%
- Reduced overfitting gap from 16% to 8.8%