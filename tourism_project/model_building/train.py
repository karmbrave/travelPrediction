import os
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

def train_model():
    # Load local splits
    print("Loading data splits...")
    X_train = pd.read_csv("Xtrain.csv")
    X_test = pd.read_csv("Xtest.csv")
    y_train = pd.read_csv("ytrain.csv").values.ravel()
    y_test = pd.read_csv("ytest.csv").values.ravel()

    # Identify numeric and categorical columns
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    # If 'Unnamed: 0' snuck in as an index column, drop it from numerical scaling
    if 'Unnamed: 0' in num_cols:
        num_cols.remove('Unnamed: 0')
        X_train = X_train.drop(columns=['Unnamed: 0'])
        X_test = X_test.drop(columns=['Unnamed: 0'])

    print(f"Numerical columns: {num_cols}")
    print(f"Categorical columns: {cat_cols}")

    # Build Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )

    # Create ML pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
    ])

    # Hyperparameter Grid Search
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.1, 0.2]
    }

    print("Starting Grid Search...")
    # Updated scoring to 'f1' to balance classes better
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)

    # Start MLflow Run
    mlflow.set_experiment("Tourism_Package_Prediction")
    with mlflow.start_run():
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        print(f"Best Hyperparameters: {best_params}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test F1 Score (weighted): {report['weighted avg']['f1-score']:.4f}")

        # Log to MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_f1_score", report['weighted avg']['f1-score'])
        mlflow.sklearn.log_model(best_model, "model")
        
        # Save model locally for Streamlit deployment
        os.makedirs("tourism_project/deployment", exist_ok=True)
        model_path = "tourism_project/deployment/best_model.pkl"
        joblib.dump(best_model, model_path)
        print(f"Model saved locally to {model_path}")

if __name__ == "__main__":
    train_model()
