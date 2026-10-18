import os
import pandas as pd

def register_data():
    data_path = "tourism_project/data/tourism.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
        
    print("Reading dataset for registration...")
    df = pd.read_csv(data_path)
    
    print("--- Dataset Summary ---")
    print(f"Total Rows: {df.shape[0]}")
    print(f"Total Columns: {df.shape[1]}")
    print("Columns Present:")
    print(list(df.columns))
    print("-----------------------")

if __name__ == '__main__':
    register_data()
