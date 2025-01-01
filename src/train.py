import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import numpy as np

# Load the datasets
bmi_df = pd.read_csv('data/bmi.csv')
meals_df = pd.read_csv('data/mealplans.csv')
nutrition_df = pd.read_csv('data/nutrition.csv')

# Clean and preprocess the BMI data
bmi_df.dropna(inplace=True)
bmi_df['Bmi'] = bmi_df['Weight'] / (bmi_df['Height'] ** 2)

# Normalize the nutritional data
columns_to_normalize = ['calories', 'total_fat', 'cholesterol', 'sodium', 'fiber', 'protein']

def extract_numeric(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return value
    return float(''.join(filter(str.isdigit, str(value))))

for col in columns_to_normalize:
    nutrition_df[col] = nutrition_df[col].apply(extract_numeric)

# Normalize the data using StandardScaler
scaler = StandardScaler()
nutrition_df[columns_to_normalize] = scaler.fit_transform(nutrition_df[columns_to_normalize])

# Train a KMeans clustering model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(nutrition_df[columns_to_normalize])

# Save the model to a file
joblib.dump(kmeans, 'models/model.pkl')