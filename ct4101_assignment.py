"""
CT4101 Assignment 1: Helper Functions
This module provides helper functions for machine learning analysis on wildfire prediction dataset.
Functions can be imported and used in Jupyter notebooks for comprehensive analysis.
"""

from __future__ import annotations

from pathlib import Path
from warnings import filterwarnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Configuration constants
DATA_DIR = Path('.')
TRAIN_PATH = DATA_DIR / 'wildfires_training.csv'
TEST_PATH = DATA_DIR / 'wildfires_test.csv'
RANDOM_STATE = 42  # For reproducible results

# Ignore the harmless warning about penalty=None skipping C parameter
filterwarnings('ignore', message='Setting penalty=None')


def load_and_prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare the wildfire dataset for training and testing.
    
    Returns:
        Tuple containing (train_df, test_df, y_train, y_test, X_train, X_test)
    """
    # Load training and test datasets
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Prepare features and target variables
    # Convert 'yes'/'no' to 1/0 for binary classification
    y_train = train_df['fire'].map({'yes': 1, 'no': 0}).astype(int)
    X_train = train_df.drop(columns='fire')
    y_test = test_df['fire'].map({'yes': 1, 'no': 0}).astype(int)
    X_test = test_df.drop(columns='fire')
    
    return train_df, test_df, y_train, y_test, X_train, X_test


def train_baseline_random_forest(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[RandomForestClassifier, float, float, float]:
    """
    Train a baseline Random Forest model with default parameters.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Tuple containing (model, train_acc, test_acc, overfitting_gap)
    """
    rf_baseline = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_baseline.fit(X_train, y_train)
    
    train_acc = rf_baseline.score(X_train, y_train)
    test_acc = rf_baseline.score(X_test, y_test)
    overfitting_gap = train_acc - test_acc
    
    return rf_baseline, train_acc, test_acc, overfitting_gap


def hyperparameter_tuning_random_forest(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Perform hyperparameter tuning for Random Forest using only 2 hyperparameters.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        DataFrame with results for all hyperparameter combinations
    """
    # Define hyperparameter ranges (only 2 hyperparameters as per assignment requirements)
    max_depth_list = [3, 4, 5, 6, 8, 10]
    min_samples_leaf_list = [1, 2, 3, 4, 5]
    
    results = []
    
    for depth in max_depth_list:
        for min_leaf in min_samples_leaf_list:
            # Train Random Forest with current hyperparameter combination
            rf = RandomForestClassifier(
                n_estimators=200,        # Fixed number of trees
                max_depth=depth,         # Variable hyperparameter 1
                min_samples_leaf=min_leaf,  # Variable hyperparameter 2
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)
            
            # Calculate performance metrics and overfitting gap
            train_acc = rf.score(X_train, y_train)
            test_acc = rf.score(X_test, y_test)
            overfitting_gap = train_acc - test_acc
            
            results.append({
                'max_depth': depth,
                'min_samples_leaf': min_leaf,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'overfitting_gap': overfitting_gap,
            })
    
    return pd.DataFrame(results)


def get_best_random_forest_model(results_df: pd.DataFrame, X_train: pd.DataFrame, y_train: pd.Series) -> tuple[RandomForestClassifier, dict]:
    """
    Get the best Random Forest model based on hyperparameter tuning results.
    
    Args:
        results_df: DataFrame with hyperparameter tuning results
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Tuple containing (best_model, best_params)
    """
    # Sort by test accuracy (descending) then by overfitting gap (ascending)
    results_sorted = results_df.sort_values(['test_acc', 'overfitting_gap'], ascending=[False, True]).reset_index(drop=True)
    
    # Select best performing model
    best_params = results_sorted.iloc[0]
    
    # Train the best model
    best_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=int(best_params['max_depth']),
        min_samples_leaf=int(best_params['min_samples_leaf']),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    best_model.fit(X_train, y_train)
    
    return best_model, best_params


def evaluate_random_forest_model(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> tuple:
    """
    Evaluate Random Forest model and return predictions and metrics.
    
    Args:
        model: Trained Random Forest model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Tuple containing (predictions, classification_report, confusion_matrix)
    """
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    
    return predictions, report, cm


def get_feature_importance(model: RandomForestClassifier, feature_names: list[str]) -> pd.DataFrame:
    """
    Get feature importance from Random Forest model.
    
    Args:
        model: Trained Random Forest model
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importance sorted by importance
    """
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return feature_importance_df