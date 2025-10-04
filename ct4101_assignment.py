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
RANDOM_STATE = 42

# Ignore the harmless message that says penalty=None skips C.
filterwarnings('ignore', message='Setting penalty=None')

def main() -> None:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    y_train = train_df['fire'].map({'yes': 1, 'no': 0}).astype(int)
    X_train = train_df.drop(columns='fire')
    y_test = test_df['fire'].map({'yes': 1, 'no': 0}).astype(int)
    X_test = test_df.drop(columns='fire')

    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)

    print("=== Logistic Regression: Default Parameters ===")
    base_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    base_model.fit(Xtr, y_train)
    base_train = base_model.predict(Xtr)
    base_test = base_model.predict(Xte)
    print(f"Train accuracy: {accuracy_score(y_train, base_train):.4f}")
    print(f"Test accuracy : {accuracy_score(y_test, base_test):.4f}")
    print("\nClassification report (test set):")
    print(classification_report(y_test, base_test))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, base_test))

    print("\n=== Logistic Regression: Manual Hyperparameter Tuning ===")
    rows: list[dict[str, float | str]] = []
    for penalty in ('l2', None):
        c_values = (1.0,) if penalty is None else (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30)
        for C in c_values:
            model = LogisticRegression(
                penalty=penalty,
                C=C,
                solver='lbfgs',
                max_iter=2000,
                random_state=RANDOM_STATE,
            )
            model.fit(Xtr, y_train)
            train_pred = model.predict(Xtr)
            test_pred = model.predict(Xte)
            rows.append(
                {
                    'penalty': 'none' if penalty is None else penalty,
                    'C': float(C),
                    'train_acc': accuracy_score(y_train, train_pred),
                    'test_acc': accuracy_score(y_test, test_pred),
                }
            )

    results = pd.DataFrame(rows).sort_values(['penalty', 'C']).reset_index(drop=True)
    print(results)
    results.to_csv('logreg_results.csv', index=False)

    best = results.loc[results['test_acc'].idxmax()]
    print("\n[Best grid-search model]")
    print(best)

    best_penalty = None if best['penalty'] == 'none' else best['penalty']
    best_model = LogisticRegression(
        penalty=best_penalty,
        C=float(best['C']),
        solver='lbfgs',
        max_iter=2000,
        random_state=RANDOM_STATE,
    ).fit(Xtr, y_train)
    best_test_pred = best_model.predict(Xte)
    print("\nClassification report (best model, test set):")
    print(classification_report(y_test, best_test_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, best_test_pred))
    print("\n=== Random Forest: Default Parameters ===")
    rf_baseline = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_baseline.fit(X_train, y_train)
    print(f"Train accuracy: {rf_baseline.score(X_train, y_train):.4f}")
    print(f"Test accuracy : {rf_baseline.score(X_test, y_test):.4f}")

    print("\n=== Random Forest: Manual Hyperparameter Tuning ===")
    n_estimators_list = (50, 100, 200, 400)
    max_depth_list = (6, 8, 10)
    rf_rows = []
    for depth in max_depth_list:
        for n_estimators in n_estimators_list:
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=depth,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)
            rf_rows.append(
                {
                    'max_depth': depth,
                    'n_estimators': n_estimators,
                    'train_acc': rf.score(X_train, y_train),
                    'test_acc': rf.score(X_test, y_test),
                }
            )

    rf_results = pd.DataFrame(rf_rows).sort_values(['max_depth', 'n_estimators']).reset_index(drop=True)
    print(rf_results)
    rf_results.to_csv('rf_results.csv', index=False)

    plt.figure()
    for depth, group in rf_results.groupby('max_depth'):
        group = group.sort_values('n_estimators')
        plt.plot(group['n_estimators'], group['test_acc'], marker='o', label=f"max_depth={depth}")
    plt.xscale('log')
    plt.xlabel('n_estimators (log scale)')
    plt.ylabel('Test accuracy')
    plt.title('Random Forest test accuracy vs n_estimators')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('rf_test_acc.png', dpi=150)
    plt.close()

    if __name__ == '__main__':
        main()
