from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

DATA_DIR = Path(".")
TRAIN_PATH = DATA_DIR / "wildfires_training.csv"
TEST_PATH = DATA_DIR / "wildfires_test.csv"
RANDOm_SEED = 42 # For reproducibility

def load_data(train_path: Path, test_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def split_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # column order: fire,year,temp,humidity,rainfall,drought_code,buildup_index,day,month,wind_speed
    y = df["fire"].map({"yes": 1, "no": 0}) # Convert target to binary
    X = df.drop(columns=["fire"])
    return X, y

def scale_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train) # Only fit on training data
    Xte = scaler.transform(X_test) # Transform test data
    return Xtr, Xte, scaler

def main() -> None:
    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)
    X_train, y_train = split_X_y(train_df)
    X_test, y_test = split_X_y(test_df)

    Xtr, Xte, scaler = scale_train_test(X_train, X_test)

    # default hyperparameters (scikit-learn will choose the best solver; binary classification uses liblinear/lbfgs by default)
    clf = LogisticRegression(random_state=RANDOm_SEED, max_iter=1000) # Increased max_iter to ensure convergence
    clf.fit(Xtr, y_train)

    y_pred_tr = clf.predict(Xtr)
    y_pred_te = clf.predict(Xte)

    acc_tr = accuracy_score(y_train, y_pred_tr)
    acc_te = accuracy_score(y_test, y_pred_te)

    print(f"[Default] Training accuracy: {acc_tr:.4f} | Test accuracy: {acc_te:.4f}")

if __name__ == "__main__":
    main()