from __future__ import annotations

import itertools
import pandas as pd
from pathlib import Path
from typing import Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = Path(".")
TRAIN_PATH = DATA_DIR / "wildfires_training.csv"
TEST_PATH = DATA_DIR / "wildfires_test.csv"
RANDOM_STATE = 42 # For reproducibility

# IO and preprocessing functions
def load_data(train_path: Path, test_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def split_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # column order: fire,year,temp,humidity,rainfall,drought_code,buildup_index,day,month,wind_speed
    y = df["fire"].map({"yes": 1, "no": 0}).astype(int) # binary target
    X = df.drop(columns=["fire"])
    return X, y

def scale_train_test(
        X_train: pd.DataFrame, X_test: pd.DataFrame
        ) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train) # Only fit on training data
    Xte = scaler.transform(X_test) # Transform test data
    return Xtr, Xte, scaler

# Manual tunning
def evaluate_logreg_grid(
        Xtr, ytr, Xte, yte,
        C_list=(0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30),
        penalty_list=("l2", None),
        solver="lbfgs",
        random_state=RANDOM_STATE
) -> pd.DataFrame:
    """
    Traverse C and penalty, return train/test accuracy for each param combo.
    Note: penalty=None (recorded as 'none') works with lbfgs/newton-cg/sag/saga (sklearn>=0.22).
    """
    rows = []
    for C, pen in itertools.product(C_list, penalty_list):
        clf = LogisticRegression(
            penalty=pen,
            C=C,
            solver=solver,
            max_iter=2000,
            random_state=random_state
        )
        clf.fit(Xtr, ytr)
        yhat_tr = clf.predict(Xtr)
        yhat_te = clf.predict(Xte)
        rows.append(
            {
                "C": float(C),
                "penalty": ("none" if pen is None else pen),
                "train_acc": accuracy_score(ytr, yhat_tr),
                "test_acc": accuracy_score(yte, yhat_te)
            }
        )

    df = pd.DataFrame(rows).sort_values(["penalty", "C"]).reset_index(drop=True)
    return df

def plot_C_vs_acc(df: pd.DataFrame, penalty: str, save_path: Path | None = None) -> None:
    sub = df[df["penalty"] == penalty]
    xs = sub["C"].values.astype(float)
    plt.figure()
    plt.plot(xs, sub["train_acc"].values, marker="o", label="train")
    plt.plot(xs, sub["test_acc"].values, marker="o", label="test")
    plt.xscale("log")
    plt.xlabel("C (log scale)")
    plt.ylabel("Accuracy")
    plt.title(f"Logistic Regression accuracy vs C (penalty={penalty})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def run_default_demo() -> None:
    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)
    X_train, y_train = split_X_y(train_df)
    X_test, y_test = split_X_y(test_df)
    Xtr, Xte, _ = scale_train_test(X_train, X_test)

    clf = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    clf.fit(Xtr, y_train)

    y_pred_tr = clf.predict(Xtr)
    y_pred_te = clf.predict(Xte)
    acc_tr = accuracy_score(y_train, y_pred_tr)
    acc_te = accuracy_score(y_test, y_pred_te)
    print(f"[Default] Train acc = {acc_tr:.4f} | Test acc = {acc_te:.4f}")

    # Run a quick grid search for comparison and visualization
    results = evaluate_logreg_grid(Xtr, y_train, Xte, y_test)
    plot_C_vs_acc(results, penalty="l2")
    plot_C_vs_acc(results, penalty="none")

    best_row = results.loc[results["test_acc"].idxmax()]
    print("\n[Best on test]")
    print(best_row)

def run_tuning_demo() -> None:
    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)
    X_train, y_train = split_X_y(train_df)
    X_test, y_test = split_X_y(test_df)
    Xtr, Xte, _ = scale_train_test(X_train, X_test)

    results = evaluate_logreg_grid(
        Xtr,
        y_train,
        Xte,
        y_test,
        C_list=(0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30),
        penalty_list=("l2", None),
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    print(results)

    # Save table for report
    results.to_csv("logreg_results.csv", index=False)

    # Plot & save
    plot_C_vs_acc(results, penalty="l2", save_path=Path("logreg_C_vs_acc_l2.png"))
    plot_C_vs_acc(results, penalty="none", save_path=Path("logreg_C_vs_acc_none.png"))

    # Best row
    best_row = results.loc[results["test_acc"].idxmax()]
    print("\n[Best on test]")
    print(best_row)


def main() -> None:
    print("=== Logistic Regression: Default Parameters ===")
    run_default_demo()

    print("\n=== Logistic Regression: Manual Hyperparameter Tuning ===")
    run_tuning_demo()


if __name__ == "__main__":
    main()