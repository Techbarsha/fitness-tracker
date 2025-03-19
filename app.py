# app.py
import streamlit as st
import joblib
import pandas as pd

# Load models
try:
    rf_regressor = joblib.load('rf_regressor.pkl')
    svm_classifier = joblib.load('svm_classifier.pkl')
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# Recommendation function
def get_recommendation(user_data):
    try:
        calorie_pred = rf_regressor.predict(user_data)
        intensity_pred = svm_classifier.predict(user_data)
        
        recommendations = []
        if calorie_pred < 300:
            recommendations.append("Consider increasing workout duration by 15 minutes")
        if intensity_pred[0] == 'Low':
            recommendations.append("Try high-intensity interval training (HIIT)")
            
        return {
            'predicted_calories': round(calorie_pred[0], 2),
            'recommended_intensity': intensity_pred[0],
            'personalized_recommendations': recommendations
        }
    except Exception as e:
        return {'error': str(e)}

# App interface
st.title('🏋️ Personal Fitness Tracker')
st.markdown("### Get personalized fitness recommendations based on your metrics")

with st.form("user_inputs"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input('Age', 18, 100, 30)
        weight = st.number_input('Weight (kg)', 40.0, 200.0, 70.0)
        height = st.number_input('Height (cm)', 100.0, 250.0, 170.0)
    with col2:
        exercise_duration = st.number_input('Exercise Duration (minutes)', 0, 180, 45)
        heart_rate = st.number_input('Heart Rate (bpm)', 60, 200, 120)
        steps = st.number_input('Steps', 0, 30000, 8000)
    activity_type = st.selectbox('Activity Type', ['Running', 'Cycling', 'Swimming', 'Weight Training'])
    
    submitted = st.form_submit_button("Get Recommendations")

if submitted:
    user_data = pd.DataFrame({
        'Age': [age],
        'Weight': [weight],
        'Height': [height],
        'Exercise_Duration': [exercise_duration],
        'Heart_Rate': [heart_rate],
        'Steps': [steps],
        'BMI': [weight / ((height/100) ** 2)],
        'Gender_Female': [0],
        'Gender_Male': [1],
        'Activity_Type_Cycling': [1 if activity_type == 'Cycling' else 0],
        'Activity_Type_Running': [1 if activity_type == 'Running' else 0],
        'Activity_Type_Swimming': [1 if activity_type == 'Swimming' else 0],
        'Activity_Type_Weight Training': [1 if activity_type == 'Weight Training' else 0]
    })
    
    results = get_recommendation(user_data)
    
    if 'error' in results:
        st.error(f"Prediction error: {results['error']}")
    else:
        st.success("Here are your personalized recommendations:")
        with st.expander("See detailed results"):
            st.metric("Predicted Calories Burned", f"{results['predicted_calories']} kcal")
            st.write(f"**Recommended Intensity:** {results['recommended_intensity']}")
            
            st.markdown("**Actionable Recommendations:**")
            for rec in results['personalized_recommendations']:
                st.write(f"✅ {rec}")

st.markdown("---")
st.info("Note: This app uses machine learning models trained on synthetic data. For real-world use, consult a fitness professional.")
