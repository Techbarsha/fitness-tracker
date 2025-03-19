import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

# Generate synthetic data for calorie prediction
np.random.seed(42)
n_samples = 1000
calories_data = {
    'age': np.random.randint(18, 70, n_samples),
    'weight': np.random.randint(50, 120, n_samples),
    'height': np.random.randint(150, 200, n_samples),
    'gender': np.random.choice([0, 1], n_samples),
    'duration': np.random.randint(10, 120, n_samples),
    'heart_rate': np.random.randint(60, 200, n_samples),
    'steps': np.random.randint(0, 20000, n_samples),
    'calories': 50 + 0.1*np.random.randint(18,70,n_samples) + 0.8*np.random.randint(50,120,n_samples) 
               + 2*np.random.randint(10,120,n_samples) + np.random.normal(0, 50, n_samples)
}
calories_df = pd.DataFrame(calories_data)

# Generate synthetic data for exercise recommendation
exercise_mapping = {0: 'Running', 1: 'Weight Training', 2: 'HIIT', 3: 'Yoga'}
exercise_data = {
    'age': np.random.randint(18, 70, 500),
    'weight': np.random.randint(50, 120, 500),
    'fitness_level': np.random.choice([0,1,2], 500),
    'goal': np.random.choice([0,1], 500),
    'preferred_type': np.random.choice([0,1,2], 500),
    'exercise': np.random.choice([0,1,2,3], 500)
}
exercise_df = pd.DataFrame(exercise_data)

# Train calorie prediction model
X_cal = calories_df.drop('calories', axis=1)
y_cal = calories_df['calories']
X_train_cal, X_test_cal, y_train_cal, y_test_cal = train_test_split(X_cal, y_cal, test_size=0.2)
cal_model = RandomForestRegressor()
cal_model.fit(X_train_cal, y_train_cal)

# Train exercise recommendation model
X_ex = exercise_df.drop('exercise', axis=1)
y_ex = exercise_df['exercise']
X_train_ex, X_test_ex, y_train_ex, y_test_ex = train_test_split(X_ex, y_ex, test_size=0.2)
ex_model = RandomForestClassifier()
ex_model.fit(X_train_ex, y_train_ex)

# Streamlit UI
st.title('🏋️ Personal Fitness Tracker')

# Calorie Prediction Section
st.header('Calorie Burn Prediction')
col1, col2 = st.columns(2)
with col1:
    age = st.number_input('Age', 18, 70, 30)
    weight = st.number_input('Weight (kg)', 50, 120, 70)
    height = st.number_input('Height (cm)', 150, 200, 170)
with col2:
    gender = st.selectbox('Gender', ['Male', 'Female'])
    duration = st.number_input('Exercise Duration (mins)', 10, 120, 30)
    heart_rate = st.number_input('Heart Rate (bpm)', 60, 200, 80)
    steps = st.number_input('Daily Steps', 0, 20000, 5000)

gender_map = {'Male': 1, 'Female': 0}
input_cal = pd.DataFrame([[
    age, weight, height, gender_map[gender], duration, heart_rate, steps
]], columns=X_cal.columns)

cal_pred = cal_model.predict(input_cal)[0]
st.subheader(f'🔥 Predicted Calories Burned: {cal_pred:.0f}')

# Exercise Recommendation Section
st.header('Exercise Recommendation')
col3, col4 = st.columns(2)
with col3:
    fitness_level = st.selectbox('Fitness Level', ['Beginner', 'Intermediate', 'Advanced'])
    goal = st.selectbox('Goal', ['Weight Loss', 'Muscle Gain'])
with col4:
    preferred_type = st.selectbox('Preferred Type', ['Cardio', 'Strength', 'Flexibility'])

input_ex = pd.DataFrame([[
    age, weight, 
    ['Beginner', 'Intermediate', 'Advanced'].index(fitness_level),
    ['Weight Loss', 'Muscle Gain'].index(goal),
    ['Cardio', 'Strength', 'Flexibility'].index(preferred_type)
]], columns=X_ex.columns)

ex_pred = ex_model.predict(input_ex)[0]
st.subheader(f'✅ Recommended Exercise: {exercise_mapping[ex_pred]}')

# Model Performance
with st.expander('Model Metrics'):
    st.write('**Calorie Model:**')
    preds = cal_model.predict(X_test_cal)
    st.metric('R² Score', f'{r2_score(y_test_cal, preds):.2f}')
    
    st.write('**Exercise Model:**')
    preds = ex_model.predict(X_test_ex)
    st.metric('Accuracy', f'{accuracy_score(y_test_ex, preds):.2f}')

# Progress Visualization
st.header('Weekly Progress')
progress = pd.DataFrame({
    'Week': [1, 2, 3, 4],
    'Calories': [2200, 2400, 2300, 2500],
    'Weight': [70, 69, 68, 67]
})
st.line_chart(progress.set_index('Week'))

st.write('*Note: Synthetic data used for demonstration purposes*')
