import os
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Load data
train_df = pd.read_csv("Kriteria 2/data_preprocessing/train_preprocessed.csv")
test_df = pd.read_csv("Kriteria 2/data_preprocessing/test_preprocessed.csv")
X_train = train_df.drop("Class", axis=1)
y_train = train_df["Class"]
X_test = test_df.drop("Class", axis=1)
y_test = test_df["Class"]


# Dagshub
dagshub.init(repo_owner='ghinaazi', repo_name='Eksperimen_SML_Ghina', mlflow=True)
mlflow.set_experiment("breast_cancer_tuning")

# Lokal
# mlflow.set_tracking_uri("http://127.0.0.1:5000/")
# mlflow.set_experiment("breast_cancer_modelling_tuning")

# Hyperparameter tuning
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy"
)


# Manual logging
with mlflow.start_run():

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("cv_best_accuracy", grid_search.best_score_)
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    metrics_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "cv_best_accuracy": grid_search.best_score_
    }

    with open("metric_info.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)

    mlflow.log_artifact("metric_info.json")


    # Confussion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    mlflow.log_artifact("confusion_matrix.png")

    # Classification report
    report = classification_report(y_test, y_pred)

    with open("classification_report.txt", "w") as f:
        f.write(report)

    mlflow.log_artifact("classification_report.txt")

    # Log model
    mlflow.sklearn.log_model(best_model, "random_forest_model")

    print("Best Parameters:", grid_search.best_params_)
    print("CV Accuracy:", grid_search.best_score_)
    print("Test Accuracy:", accuracy)
