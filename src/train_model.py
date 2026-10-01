import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

# Load dataset
data_path = os.path.join("..", "data", "Heart_Disease_Prediction.csv")
df = pd.read_csv(data_path)

# Rename columns for convenience
df.columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']

# Features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model trained! Test Accuracy: {accuracy*100:.2f}%")

# Save model
model_path = os.path.join("..", "models", "heart_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"Model saved at {model_path}")
