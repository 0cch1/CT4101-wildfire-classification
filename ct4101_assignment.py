from __future__ import annotations

from pathlib import Path
from warnings import filterwarnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path('.')
TRAIN_PATH = DATA_DIR / 'wildfires_training.csv'
TEST_PATH = DATA_DIR / 'wildfires_test.csv'
RANDOM_STATE = 42  # For reproducible results

# Ignore the harmless warning about penalty=None skipping C parameter
filterwarnings('ignore', message='Setting penalty=None')

def main() -> None:
    """
    Main function that loads data, performs model training and evaluation.
    Compares Logistic Regression and Random Forest with hyperparameter tuning.
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

    # Standardize features for Logistic Regression (Random Forest doesn't need scaling)
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)  # Scaled features for Logistic Regression
    Xte = scaler.transform(X_test)

    # === LOGISTIC REGRESSION EXPERIMENTS ===
    
    print("=== Logistic Regression: Default Parameters ===")
    # Train baseline Logistic Regression model with default parameters
    base_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    base_model.fit(Xtr, y_train)
    base_train = base_model.predict(Xtr)
    base_test = base_model.predict(Xte)
    
    # Display performance metrics    print(f"Train accuracy: {accuracy_score(y_train, base_train):.4f}")
    print(f"Test accuracy : {accuracy_score(y_test, base_test):.4f}")
    print("\nClassification report (test set):")
    print(classification_report(y_test, base_test))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, base_test))

    print("\n=== Logistic Regression: Manual Hyperparameter Tuning ===")
    # Grid search for optimal hyperparameters
    rows: list[dict[str, float | str]] = []
    # Test different penalty types: L2 regularization and no regularization
    for penalty in ('l2', None):
        # C values: smaller = more regularization, larger = less regularization
        # When penalty=None, C parameter is ignored, so we use only C=1.0
        c_values = (1.0,) if penalty is None else (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30)
        for C in c_values:
            model = LogisticRegression(
                penalty=penalty,
                C=C,
                solver='lbfgs',  # Good for small datasets
                max_iter=2000,
                random_state=RANDOM_STATE,
            )
            model.fit(Xtr, y_train)
            train_pred = model.predict(Xtr)
            test_pred = model.predict(Xte)            
            # Store results for comparison
            rows.append(
                {
                    'penalty': 'none' if penalty is None else penalty,
                    'C': float(C),
                    'train_acc': accuracy_score(y_train, train_pred),
                    'test_acc': accuracy_score(y_test, test_pred),
                }
            )

    # Analyze results and find best performing model
    results = pd.DataFrame(rows).sort_values(['penalty', 'C']).reset_index(drop=True)
    print(results)
    results.to_csv('logreg_results.csv', index=False)

    # Select model with highest test accuracy
    best = results.loc[results['test_acc'].idxmax()]
    print("\n[Best grid-search model]")
    print(best)

    # Retrain best model for detailed evaluation
    best_penalty = None if best['penalty'] == 'none' else best['penalty']
    best_model = LogisticRegression(
        penalty=best_penalty,
        C=float(best['C']),
        solver='lbfgs',
        max_iter=2000,
        random_state=RANDOM_STATE,
    ).fit(Xtr, y_train)
    best_test_pred = best_model.predict(Xte)
    
    # Display detailed performance metrics for best model
    print("\nClassification report (best model, test set):")
    print(classification_report(y_test, best_test_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, best_test_pred))
    # === RANDOM FOREST EXPERIMENTS ===
    
    print("\n=== Random Forest: Default Parameters ===")
    # Train baseline Random Forest model (no feature scaling needed)
    rf_baseline = RandomForestClassifier(
        n_estimators=200,    # Number of trees in the forest
        max_depth=8,         # Maximum depth of each tree
        random_state=RANDOM_STATE,
        n_jobs=-1,           # Use all available cores
    )
    rf_baseline.fit(X_train, y_train)  # Use unscaled features
    print(f"Train accuracy: {rf_baseline.score(X_train, y_train):.4f}")
    print(f"Test accuracy : {rf_baseline.score(X_test, y_test):.4f}")

    print("\n=== Random Forest: Manual Hyperparameter Tuning ===")
    # Define hyperparameter ranges for grid search
    n_estimators_list = (50, 100, 200)           # Number of trees
    max_depth_list = (3, 4, 5, 6)                # Tree depth (prevent overfitting)
    min_samples_split_list = (2, 5, 10)          # Minimum samples to split a node
    min_samples_leaf_list = (1, 2, 4)            # Minimum samples per leaf
    max_features_list = ('sqrt', 0.5, 0.3)       # Features per split (reduce overfitting)
    
    # Perform grid search over all hyperparameter combinations
    rf_rows = []
    for depth in max_depth_list:
        for n_estimators in n_estimators_list:
            for min_split in min_samples_split_list:
                for min_leaf in min_samples_leaf_list:
                    for max_feat in max_features_list:
                        # Train Random Forest with current hyperparameter combination
                        rf = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=depth,
                            min_samples_split=min_split,
                            min_samples_leaf=min_leaf,
                            max_features=max_feat,
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        )
                        rf.fit(X_train, y_train)
                        
                        # Calculate performance metrics and overfitting gap
                        rf_rows.append(
                            {
                                'max_depth': depth,
                                'n_estimators': n_estimators,
                                'min_samples_split': min_split,
                                'min_samples_leaf': min_leaf,
                                'max_features': str(max_feat),
                                'train_acc': rf.score(X_train, y_train),
                                'test_acc': rf.score(X_test, y_test),
                                'overfitting_gap': rf.score(X_train, y_train) - rf.score(X_test, y_test),
                            }
                        )

    # Analyze results and select best model
    # Sort by test accuracy (descending) then by overfitting gap (ascending)
    rf_results = pd.DataFrame(rf_rows).sort_values(['test_acc', 'overfitting_gap'], ascending=[False, True]).reset_index(drop=True)
    print(rf_results.head(10))
    
    # Select best performing model
    best_rf = rf_results.iloc[0]
    print(f"\n[Best model]")
    print(f"Parameters: max_depth={best_rf['max_depth']}, n_estimators={best_rf['n_estimators']}")
    print(f"Train accuracy: {best_rf['train_acc']:.4f}")
    print(f"Test accuracy: {best_rf['test_acc']:.4f}")
    
    # Retrain best model for detailed evaluation
    best_rf_model = RandomForestClassifier(
        n_estimators=int(best_rf['n_estimators']),
        max_depth=int(best_rf['max_depth']),
        min_samples_split=int(best_rf['min_samples_split']),
        min_samples_leaf=int(best_rf['min_samples_leaf']),
        max_features=float(best_rf['max_features']) if best_rf['max_features'] not in ['sqrt', 'log2'] else best_rf['max_features'],
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ).fit(X_train, y_train)
    
    # Generate predictions and detailed performance metrics
    best_rf_pred = best_rf_model.predict(X_test)
    print("\nClassification report (best model, test set):")
    print(classification_report(y_test, best_rf_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, best_rf_pred))
    
    # Save results to CSV file
    rf_results.to_csv('rf_results.csv', index=False)

if __name__ == '__main__':
    main()
