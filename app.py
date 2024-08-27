
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

app = Flask(__name__)


# Load the datasets
bmi_df = pd.read_csv('csv/bmi.csv')
meals_df = pd.read_csv('csv/mealplans.csv')
nutrition_df = pd.read_csv('csv/nutrition.csv')

# Clean and preprocess the BMI data
bmi_df.dropna(inplace=True)
bmi_df['Bmi'] = bmi_df['Weight'] / (bmi_df['Height'] ** 2)
bmi_df['BmiClass'] = pd.cut(bmi_df['Bmi'], bins=[0, 18.5, 24.9, 29.9, 34.9, 39.9, np.inf],
                            labels=['Underweight', 'Normal weight', 'Overweight', 'Obese Class 1', 'Obese Class 2', 'Obese Class 3'])

# Normalize the nutritional data
def extract_numeric(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return value
    return float(''.join(filter(str.isdigit, str(value))))

columns_to_normalize = ['calories', 'total_fat', 'cholesterol', 'sodium', 'fiber', 'protein']

for col in columns_to_normalize:
    if col not in nutrition_df.columns:
        print(f"Column {col} not found in the dataset")
    else:
        nutrition_df[col] = nutrition_df[col].apply(extract_numeric)
        nutrition_df[col] = pd.to_numeric(nutrition_df[col], errors='coerce')

# Remove rows with NaN values after preprocessing
nutrition_df.dropna(subset=columns_to_normalize, inplace=True)

scaler = StandardScaler()
nutrition_normalized = scaler.fit_transform(nutrition_df[columns_to_normalize])
nutrition_normalized_df = pd.DataFrame(nutrition_normalized, columns=columns_to_normalize)

# Perform clustering on the nutrition data
kmeans = KMeans(n_clusters=5, random_state=42)
nutrition_df['cluster'] = kmeans.fit_predict(nutrition_normalized_df)

# Generate meal plan based on BMI
def generate_meal_plan(bmi_class, num_days):
    if bmi_class == 'Underweight':
        cluster = 0
    elif bmi_class == 'Normal weight':
        cluster = 1
    elif bmi_class == 'Overweight':
        cluster = 2
    elif bmi_class == 'Obese Class 1':
        cluster = 3
    else:
        cluster = 4
    meal_plan = nutrition_df[nutrition_df['cluster'] == cluster].sample(num_days)
    meal_plan.reset_index(drop=True, inplace=True)
    meal_plan['Day'] = range(1, num_days + 1)
    return meal_plan[['Day', 'name', 'calories', 'total_fat', 'protein', 'sodium', 'fiber']]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        weight = float(request.form['weight'])
        height_cm = float(request.form['height'])
        height_m = height_cm / 100  # convert height from cm to m
        bmi = weight / (height_m ** 2)
        if bmi < 18.5:
            bmi_class = 'Underweight'
        elif bmi < 24.9:
            bmi_class = 'Normal weight'
        elif bmi < 29.9:
            bmi_class = 'Overweight'
        elif bmi < 34.9:
            bmi_class = 'Obese Class 1'
        else:
            bmi_class = 'Obese Class 2'
        meal_plan = generate_meal_plan(bmi_class, 30)
        return render_template('meal_plan.html', bmi=bmi, bmi_class=bmi_class, meal_plan=meal_plan)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)