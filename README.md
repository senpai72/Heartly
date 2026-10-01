# Heart Disease Prediction App

A machine learning web application that predicts whether a patient is at risk of heart disease based on medical attributes such as age, sex, blood pressure, cholesterol, chest pain type, and other heart-related indicators.

## Overview

This project combines:

- A machine learning model trained on a heart disease dataset
- A Streamlit web interface for user input
- Real-time prediction output showing whether the patient is at low or high risk

The model is trained using a Random Forest Classifier and saved as a pickle file so the web app can load it and make predictions quickly.

## Project Structure

- `data/` – dataset used for training the model
- `models/` – trained model file (`heart_model.pkl`)
- `src/` – application and training scripts
  - `train_model.py` – loads the dataset, trains the model, evaluates accuracy, and saves the model
  - `app.py` – Streamlit app for collecting patient information and displaying predictions
- `requirements.txt` – required Python packages

## Model Details

The project uses the following feature set to predict heart disease:

- age
- sex
- chest pain type
- resting blood pressure
- cholesterol
- fasting blood sugar
- resting ECG results
- maximum heart rate achieved
- exercise-induced angina
- ST depression
- slope of peak exercise ST segment
- number of major vessels colored by fluoroscopy
- thallium stress test result

The target label is whether heart disease is present or absent.

## How It Works

1. The dataset is loaded from the CSV file in the `data` folder.
2. The data is split into training and testing sets.
3. A Random Forest model is trained on the features.
4. The model is evaluated using accuracy.
5. The trained model is serialized and saved as a pickle file.
6. The Streamlit app loads the model and accepts patient input from the UI.
7. The app predicts whether the patient has a high or low risk of heart disease.

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the Project

### Train the model

```bash
python src/train_model.py
```

This script trains the model and saves it to the `models` directory.

### Start the app

```bash
streamlit run src/app.py
```

Then open the local URL shown in the terminal (usually http://localhost:8501).

## Notes

- The app expects the trained model file to exist in `models/heart_model.pkl`.
- If the model file is missing, train it first using `train_model.py`.
- Predictions are based on the trained dataset and should be used as an informational aid, not medical advice.

## Example Use Case

A doctor or medical analyst can enter patient details into the app to estimate the probability of heart disease risk in a quick, user-friendly interface.
