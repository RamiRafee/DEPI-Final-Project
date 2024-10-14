# data_loader.py
from zenml.steps import step
import pandas as pd
import os

@step
def load_data() -> pd.DataFrame:
    # Load the CSV file containing file paths and bounding box information
    csv_path = './Dataset/labelsEG.csv'  # Adjust this path as needed
    df = pd.read_csv(csv_path)
    
    # Update the 'filepath' column to have the correct absolute path to the Dataset folder
    df['filepath'] = df['filepath'].apply(lambda x: "./Dataset"+ x)
    print(df.head(5))
    return df