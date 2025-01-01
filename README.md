# AI Model Project

This project is designed to develop an AI model that analyzes BMI data and provides meal plans based on nutritional information. The application is built using Flask and utilizes machine learning techniques for clustering.

## Project Structure

```
BMIMODEL
├── data
│   ├── bmi.csv
│   ├── mealplans.csv
│   └── nutrition.csv
├── models
│   └── model.pkl
├── src
│   ├── app.py
│   ├── preprocess.py
│   └── train.py
├── requirements.txt
└── README.md
```

## Data Files

- **data/bmi.csv**: Contains the BMI dataset used for analysis and model training.
- **data/mealplans.csv**: Contains meal plan data that may be used in conjunction with the AI model.
- **data/nutrition.csv**: Contains nutritional information utilized for preprocessing and model training.

## Model

- **models/model.pkl**: The serialized AI model used for predictions.

## Source Code

- **src/app.py**: Main application file that sets up the Flask web server, loads datasets, preprocesses data, and serves the AI model.
- **src/preprocess.py**: Contains functions for data cleaning and preprocessing, including normalization and extraction of numeric values from the nutritional dataset.
- **src/train.py**: Contains the logic for training the AI model, including the implementation of the KMeans clustering algorithm and any necessary model evaluation.

## Requirements

To install the necessary dependencies, run:

```
pip install -r requirements.txt
```

## Usage

1. Ensure all data files are in the `data` directory.
2. Run the application using:

```
python src/app.py
```

3. Access the web application in your browser at `http://127.0.0.1:5000`.

## License

This project is licensed under the MIT License.