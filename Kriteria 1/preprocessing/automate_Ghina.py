# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump
import os



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


def automated_preprocessing(
    df: pd.DataFrame,
    target_col: str,
    save_preprocessor_path: str | None = None,
    test_size: float = 0.3,
    random_state: int = 42
):

    # 1. Cleaning awal
    df = df.dropna()
    df = df.drop_duplicates()
    
    # 2. Identifikasi kolom numerik
    target_col = 'Class'
    numerical_cols = df.select_dtypes(include='number').columns.drop(target_col)

    # 3. Remove outlier (SEBELUM split)
    iqr_remover = IQRRemover(features=numerical_cols)
    df_clean = iqr_remover.fit_transform(df)

    # 4. Split data
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 5. Preprocessing pipeline (scaling)
    numeric_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numerical_cols)
        ]
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # 6. Simpan preprocessor
    if save_preprocessor_path is not None:
        dump(preprocessor, save_preprocessor_path)

    return X_train, X_test, y_train, y_test
    

if __name__ == "__main__":
    # 1. Load dataset
    df = pd.read_csv("Kriteria 1/preprocessing/raw_dataset.csv")

    # 2. Jalankan preprocessing
    X_train, X_test, y_train, y_test = automated_preprocessing(
        df=df,
        target_col='Class',
        save_preprocessor_path='Kriteria 1/preprocessing/preprocessor.joblib'
    )

    # 3. Buat folder output jika belum ada
    output_folder = 'Kriteria 1/preprocessing/dataset_preprocessing'
    os.makedirs(output_folder, exist_ok=True)

    # 4. Simpan CSV
    output_file = os.path.join(output_folder, 'data_clean.csv')
    train_df = pd.DataFrame(X_train, columns=[col for col in df.columns if col != 'Class'])
    train_df['Class'] = y_train.reset_index(drop=True)
    train_df.to_csv(output_file, index=False)
