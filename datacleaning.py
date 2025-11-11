"""
Data Preprocessing and Analysis Pipeline
"""
import pandas as pd
import os
import glob
from sklearn.preprocessing import StandardScaler

# Load data from CSV files
path = "C:/<enter path>"
csv_files = glob.glob(os.path.join(path, "*.csv"))
all_datasets = []
print(f"Found {len(csv_files)} CSV files\n")

for id, file_path in enumerate(csv_files, start=1):
    filename = os.path.basename(file_path)
    
    try:
        df = pd.read_csv(file_path)
        df['Measurement_Point'] = id
        all_datasets.append(df)
        print(f" Loaded {filename}: {len(df)} rows (Point {id})")
    
    except FileNotFoundError:
        print(f" File not found: {filename}")
    except pd.errors.EmptyDataError:
        print(f" Empty file: {filename}")
    except Exception as e:
        print(f"Error in {filename}: {e}")

df = pd.concat(all_datasets, ignore_index=True)
print(f"\nCombined dataset: {len(df)} rows from {len(all_datasets)} measurement points")
print(df.head())


def preview_data(df):
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f" Missing values:\n{missing[missing > 0]}")
    else:
        print(" No missing values")

    print(f"\n SINR range: [{df['SINR'].min():.1f}, {df['SINR'].max():.1f}]")
    print(f"\n RSRP range: [{df['RSRP'].min():.1f}, {df['RSRP'].max():.1f}]")

def clean_data(df):
    initial_rows = len(df)
    df_clean = df.dropna()
    # extreme outliers ( values beyong SD =3 is removed)
    for col in ['SINR', 'RSRP']:
        mean = df_clean[col].mean()
        std = df_clean[col].std()
        df_clean = df_clean[
            (df_clean[col] >= mean - 3*std) & 
            (df_clean[col] <= mean + 3*std)
        ]
    removed = initial_rows - len(df_clean)
    print(f" Removed {removed} rows ({removed/initial_rows*100:.1f}%)")
    print(f" Clean dataset: {len(df_clean)} rows")
    
    return df_clean

def preprocessing(df):
    df_processed = df.copy()
    required_cols = ['Measurement_Point', 'SINR', 'RSRP']
    for col in required_cols:
        if col not in df_processed.columns:
            raise KeyError(f"Missing required column: {col}")

    numerical_features = ['SINR', 'RSRP']
    scaler = StandardScaler()
    
    df_processed[['SINR', 'RSRP']] = (df_processed.groupby('Measurement_Point')[['SINR', 'RSRP']].transform(lambda x: (x - x.mean()) / x.std()))
    df_processed['Condition'] = df_processed['Measurement_Point']
    print("Standardized features (mean ≈ 0, std ≈ 1):")
    for feature in numerical_features:
        print(f"  {feature}: mean={df_processed[feature].mean():.3f}, std={df_processed[feature].std():.3f}")
    
    print(f"\nFinal  shape: {df_processed.shape}")
        
    return df_processed, scaler

# call for function usage 
# Data Cleaning
preview_data(df)
df_clean = clean_data(df)

# preprocessing
df_cvae, scaler = preprocessing(df_clean)
df_cvae.to_csv("preprocessed_data.csv", index=False)
print(df_cvae.head())



