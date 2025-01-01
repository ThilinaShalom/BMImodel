from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def load_data(bmi_path, meals_path, nutrition_path):
    bmi_df = pd.read_csv(bmi_path)
    meals_df = pd.read_csv(meals_path)
    nutrition_df = pd.read_csv(nutrition_path)
    return bmi_df, meals_df, nutrition_df

def clean_bmi_data(bmi_df):
    bmi_df.dropna(inplace=True)
    bmi_df['Bmi'] = bmi_df['Weight'] / (bmi_df['Height'] ** 2)
    bmi_df['BmiClass'] = pd.cut(bmi_df['Bmi'], bins=[0, 18.5, 24.9, 29.9, 34.9, 39.9, np.inf],
                                labels=['Underweight', 'Normal weight', 'Overweight', 'Obese Class 1', 'Obese Class 2', 'Obese Class 3'])
    return bmi_df

def extract_numeric(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return value
    return float(''.join(filter(str.isdigit, str(value))))

def normalize_nutrition_data(nutrition_df, columns_to_normalize):
    for col in columns_to_normalize:
        if col not in nutrition_df.columns:
            print(f"Column {col} not found in the dataset")
        else:
            nutrition_df[col] = nutrition_df[col].apply(extract_numeric)
    return nutrition_df

def preprocess_data(bmi_path, meals_path, nutrition_path):
    bmi_df, meals_df, nutrition_df = load_data(bmi_path, meals_path, nutrition_path)
    bmi_df = clean_bmi_data(bmi_df)
    columns_to_normalize = ['calories', 'total_fat', 'cholesterol', 'sodium', 'fiber', 'protein']
    nutrition_df = normalize_nutrition_data(nutrition_df, columns_to_normalize)
    return bmi_df, meals_df, nutrition_df