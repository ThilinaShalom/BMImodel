from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from preprocess import extract_numeric
import joblib

app = Flask(__name__, template_folder='../templates')

# Load the datasets
bmi_df = pd.read_csv('data/bmi.csv')
meals_df = pd.read_csv('data/mealplans.csv')
nutrition_df = pd.read_csv('data/nutrition.csv')

# Clean and preprocess the BMI data
bmi_df.dropna(inplace=True)
bmi_df['Bmi'] = bmi_df['Weight'] / (bmi_df['Height'] ** 2)
bmi_df['BmiClass'] = pd.cut(bmi_df['Bmi'], bins=[0, 18.5, 24.9, 29.9, 34.9, 39.9, np.inf],
                            labels=['Underweight', 'Normal weight', 'Overweight', 'Obese Class 1', 'Obese Class 2', 'Obese Class 3'])

# Normalize the nutritional data
columns_to_normalize = ['calories', 'total_fat', 'cholesterol', 'sodium', 'fiber', 'protein']

for col in columns_to_normalize:
    if col not in nutrition_df.columns:
        print(f"Column {col} not found in the dataset")
    else:
        nutrition_df[col] = nutrition_df[col].apply(extract_numeric)

scaler = StandardScaler()
nutrition_df[columns_to_normalize] = scaler.fit_transform(nutrition_df[columns_to_normalize])

# Load the trained model
model = joblib.load('models/model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    answers = [request.form[f'question{i}'] for i in range(1, 21)]
    
    # Convert answers to a DataFrame
    user_data = pd.DataFrame([answers], columns=[f'question{i}' for i in range(1, 21)])
    
    # Example: Use only a subset of answers for clustering
    user_features = user_data[['question3', 'question4', 'question5', 'question7', 'question11']].astype(float)
    
    # Add a dummy feature to match the expected number of features
    user_features['dummy_feature'] = 0
    
    # Fit a new StandardScaler on the user features
    user_scaler = StandardScaler()
    user_features = user_scaler.fit_transform(user_features)
    
    # Predict the cluster
    cluster = model.predict(user_features)[0]
    
    # Generate plans based on the cluster
    if cluster == 0:
        workout_plan = generate_workout_plan("Cardio-focused", user_data)
        meal_plan = generate_meal_plan("High-protein", user_data)
    elif cluster == 1:
        workout_plan = generate_workout_plan("Strength training-focused", user_data)
        meal_plan = generate_meal_plan("Balanced", user_data)
    else:
        workout_plan = generate_workout_plan("Flexibility and balance-focused", user_data)
        meal_plan = generate_meal_plan("Low-carb", user_data)

    return render_template('plan.html', workout_plan=workout_plan, meal_plan=meal_plan)

def generate_workout_plan(focus, user_data):
    plan = {}
    for day in range(1, 31):
        if focus == "Cardio-focused":
            if day % 7 in [1, 4]:
                plan[day] = f"Cardio - {user_data['question6'][0]} for {user_data['question7'][0]} minutes"
            elif day % 7 in [2, 5]:
                plan[day] = "Strength Training"
            elif day % 7 == 3:
                plan[day] = "Rest"
            else:
                plan[day] = "Flexibility Exercises"
        elif focus == "Strength training-focused":
            if day % 7 in [1, 4]:
                plan[day] = "Strength Training"
            elif day % 7 in [2, 5]:
                plan[day] = f"Cardio - {user_data['question6'][0]} for {user_data['question7'][0]} minutes"
            elif day % 7 == 3:
                plan[day] = "Rest"
            else:
                plan[day] = "Flexibility Exercises"
        else:
            if day % 7 in [1, 4]:
                plan[day] = "Flexibility Exercises"
            elif day % 7 in [2, 5]:
                plan[day] = f"Cardio - {user_data['question6'][0]} for {user_data['question7'][0]} minutes"
            elif day % 7 == 3:
                plan[day] = "Rest"
            else:
                plan[day] = "Strength Training"
    return plan

def generate_meal_plan(focus, user_data):
    plan = {}
    for day in range(1, 31):
        if focus == "High-protein":
            plan[day] = {
                "breakfast": "Eggs",
                "lunch": "Chicken Salad",
                "dinner": "Grilled Fish"
            }
        elif focus == "Balanced":
            plan[day] = {
                "breakfast": "Oatmeal",
                "lunch": "Turkey Sandwich",
                "dinner": "Stir-fried Vegetables"
            }
        else:
            plan[day] = {
                "breakfast": "Smoothie",
                "lunch": "Salad",
                "dinner": "Grilled Chicken"
            }
    return plan

if __name__ == '__main__':
    app.run(debug=True)