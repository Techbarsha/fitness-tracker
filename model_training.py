# model_training.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
import joblib

# Generate synthetic data (same as before)
def generate_fitness_data(num_samples=1000):
    # ... [same data generation code as previous implementation] ...
    return pd.DataFrame(data)

# Create and preprocess data
df = generate_fitness_data()
df['BMI'] = df['Weight'] / ((df['Height']/100) ** 2)
df = pd.get_dummies(df, columns=['Gender', 'Activity_Type'])

# Prepare data
X = df.drop(['Calories_Burned', 'Workout_Intensity'], axis=1)
y_reg = df['Calories_Burned']
y_clf = df['Workout_Intensity']

# Train models
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X, y_reg)

svm_classifier = SVC(kernel='rbf', probability=True)
svm_classifier.fit(X, y_clf)

# Save models
joblib.dump(rf_regressor, 'rf_regressor.pkl')
joblib.dump(svm_classifier, 'svm_classifier.pkl')

print("Models saved successfully!")
