# -*- coding: utf-8 -*-

import os
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump


# ===============================
# Custom Transformer: IQR Remover
# ===============================
class IQRRemover(BaseEstimator, TransformerMixin):
    def __init__(self, features, factor=1.5):
        self.features = features
        self.factor = factor

    def fit(self, X, y=None):
        self.Q1_ = X[self.features].quantile(0.25)
        self.Q3_ = X[self.features].quantile(0.75)
        self.IQR_ = self.Q3_ - self.Q1_
        return self

    def transform(self, X):
        mask = ~(
            (X[self.features] < (self.Q1_ - self.factor * self.IQR_)) |
            (X[self.features] > (self.Q3_ + self.factor * self.IQR_))
        ).any(axis=1)
        return X.loc[mask]


# ===============================
# Automated Preprocessing Function
# ===============================
def automated_preprocessing(
    df: pd.DataFrame,
    save_preprocessor_path: str | None = None,
    test_size: float = 0.3,
    random_state: int = 42
):
    # 1. Cleaning awal
    target_col = 'Class'
    df = df.dropna().drop_duplicates()

    # 2. Split fitur & target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 3. Train-test split (STRATIFIED)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 4. Identifikasi kolom numerik
    numerical_cols = X_train.select_dtypes(include='number').columns.tolist()

    # 5. Outlier removal (FIT hanya di TRAIN)
    iqr_remover = IQRRemover(features=numerical_cols)
    X_train = iqr_remover.fit_transform(X_train)
    y_train = y_train.loc[X_train.index]

    X_test = iqr_remover.transform(X_test)
    y_test = y_test.loc[X_test.index]

    # 6. Preprocessing pipeline (Scaling)
    numeric_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numerical_cols)
        ],
        remainder='drop'
    )

    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    # 7. Simpan preprocessor
    if save_preprocessor_path:
        dump(preprocessor, save_preprocessor_path)

    return (
        X_train_scaled,
        X_test_scaled,
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
        numerical_cols
    )


# ===============================
# Main Execution
# ===============================
if __name__ == "__main__":

    # Load dataset
    df = pd.read_csv("raw_dataset.csv")

    # Run preprocessing
    X_train, X_test, y_train, y_test, num_cols = automated_preprocessing(
        df=df,
        target_col="Class",
        save_preprocessor_path="preprocessor.joblib"
    )

    # Output directory
    output_dir = "dataset_preprocessing"
    os.makedirs(output_dir, exist_ok=True)

    # Save TRAIN data
    train_df = pd.DataFrame(X_train, columns=num_cols)
    train_df["Class"] = y_train
    train_df.to_csv(os.path.join(output_dir, "train_preprocessed.csv"), index=False)

    # Save TEST data
    test_df = pd.DataFrame(X_test, columns=num_cols)
    test_df["Class"] = y_test
    test_df.to_csv(os.path.join(output_dir, "test_preprocessed.csv"), index=False)

    print("✅ Preprocessing selesai.")
    print(f"📁 Output disimpan di folder: {output_dir}")
