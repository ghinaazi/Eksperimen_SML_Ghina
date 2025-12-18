import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

train_path = "data_preprocessing/train_preprocessed.csv"
test_path = "data_preprocessing/test_preprocessed.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

X_train = train_df.drop("target", axis=1)
y_train = train_df["target"]

X_test = test_df.drop("target", axis=1)
y_test = test_df["target"]

# Hyperparameter tuning
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy')

# Manual logging with MLflow
with mlflow.start_run():
    # Fit model
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Log hyperparameters
    mlflow.log_params(best_params)
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Log metrics manually
    mlflow.log_metric("test_accuracy", acc)
    
    # Log model artifact
    mlflow.sklearn.log_model(best_model, "random_forest_model")
    
    print("Best Parameters:", best_params)
    print("Test Accuracy:", acc)
