# Step 4: Create Streamlit App (app.py)
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = joblib.load('calorie_model.pkl')

st.set_page_config(page_title="Fitness Tracker", layout="wide")
st.title("AI-Powered Fitness Tracker 🏋️")

# Sidebar inputs
with st.sidebar:
    st.header("Workout Details")
    steps = st.slider("Steps", 0, 20000, 10000)
    heart_rate = st.slider("Heart Rate (bpm)", 60, 200, 120)
    bmi = st.slider("BMI", 15.0, 35.0, 25.0)
    active_mins = st.slider("Active Minutes", 0, 300, 60)
    intensity = st.selectbox("Workout Intensity", ['Low', 'Moderate', 'High'])

# Encode intensity
intensity_mapping = {
    'Low': [1, 0, 0],
    'Moderate': [0, 1, 0],
    'High': [0, 0, 1]
}

# Create input dataframe
input_data = pd.DataFrame([[
    steps,
    heart_rate,
    bmi,
    active_mins,
    *intensity_mapping[intensity]
], columns=X.columns)

# Prediction and Visualization
if st.button("Calculate Calories Burned"):
    prediction = model.predict(input_data)[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Calorie Prediction 🔥")
        st.metric(label="Estimated Calories Burned", value=f"{prediction:.0f} kcal")
        
        # Progress gauge
        fig, ax = plt.subplots()
        ax.pie([prediction, 3000-prediction], 
               labels=['Burned', 'Remaining'], 
               colors=['#FF4B4B', '#DCDCDC'],
               startangle=90)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Workout Analysis 📈")
        # Intensity comparison
        intensity_data = pd.DataFrame({
            'Intensity': ['Low', 'Moderate', 'High'],
            'Your Workout': intensity_mapping[intensity],
            'Average': [0.2, 0.5, 0.3]
        })
        st.bar_chart(intensity_data.set_index('Intensity'))
        
        # Heart rate zone
        st.write("Heart Rate Zones:")
        st.progress(heart_rate/200)

# Additional Features
st.header("Fitness Dashboard")
tab1, tab2 = st.tabs(["Workout Trends", "Health Tips"])

with tab1:
    # Generate sample data
    trend_data = pd.DataFrame({
        'Day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        'Calories': [300, 450, 400, 500, 480, 600, 350]
    })
    st.line_chart(trend_data.set_index('Day'))

with tab2:
    st.subheader("Personalized Recommendations")
    if prediction > 500:
        st.success("🔥 Great intensity! Keep it up!")
        st.write("Try adding strength training 3x/week")
    else:
        st.info("💪 Good start! Try increasing workout duration by 15 minutes")
        st.write("Consider adding interval training")

# Step 5: Deployment
st.sidebar.markdown("""
**Deployment Instructions:**
1. Create `requirements.txt` with:
