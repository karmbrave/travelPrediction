import os
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data():
    # Path to the dataset
    data_path = "tourism_project/data/tourism.csv"
    
    # Check if dataset exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
        
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    
    # Remove unnecessary columns
    # CustomerID is a unique identifier and is not needed for modeling.
    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])
        print("Removed CustomerID column.")
        
    # Split into features (X) and target (y)
    # ProdTaken is the target variable indicating package purchase
    target_col = 'ProdTaken'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save files locally as CSV in the root/current working directory
    # so the GitHub Actions workflow can pass them to the next job as artifacts.
    X_train.to_csv("Xtrain.csv", index=False)
    X_test.to_csv("Xtest.csv", index=False)
    y_train.to_csv("ytrain.csv", index=False)
    y_test.to_csv("ytest.csv", index=False)
    
    print("Data split successfully and saved locally:")
    print(f"- Xtrain.csv shape: {X_train.shape}")
    print(f"- Xtest.csv shape: {X_test.shape}")
    print(f"- ytrain.csv shape: {y_train.shape}")
    print(f"- ytest.csv shape: {y_test.shape}")

if __name__ == "__main__":
    prepare_data()
