import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cache model loading
@st.cache_resource
def load_model():
    return joblib.load('calorie_model.pkl')

model = load_model()

st.set_page_config(page_title="Fitness Tracker", layout="wide")
st.title("AI-Powered Fitness Tracker 🏋️")

# Initialize prediction variable
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# Sidebar inputs
with st.sidebar:
    st.header("Workout Details")
    steps = st.slider("Steps", 0, 20000, 10000)
    heart_rate = st.slider("Heart Rate (bpm)", 60, 200, 120)
    bmi = st.slider("BMI", 15.0, 35.0, 25.0)
    active_mins = st.slider("Active Minutes", 0, 300, 60)
    intensity = st.selectbox("Workout Intensity", ['Low', 'Moderate', 'High'])

# Create input dataframe
input_data = pd.DataFrame([[
    steps,
    heart_rate,
    bmi,
    active_mins,
    1 if intensity == "Low" else 0,
    1 if intensity == "Moderate" else 0,
    1 if intensity == "High" else 0
]], columns=['Steps', 'Heart_Rate', 'BMI', 'Active_Minutes', 
           'Intensity_Low', 'Intensity_Moderate', 'Intensity_High'])

# Prediction and Visualization
if st.button("Calculate Calories Burned"):
    st.session_state.prediction = model.predict(input_data)[0]
    
if st.session_state.prediction is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Calorie Prediction 🔥")
        st.metric(label="Estimated Calories Burned", value=f"{st.session_state.prediction:.0f} kcal")
        
        # Progress gauge
        fig, ax = plt.subplots()
        ax.pie([st.session_state.prediction, 3000-st.session_state.prediction], 
               labels=['Burned', 'Remaining'], 
               colors=['#FF4B4B', '#DCDCDC'],
               startangle=90)
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.subheader("Workout Analysis 📈")
        # Intensity comparison
        intensity_data = pd.DataFrame({
            'Intensity': ['Low', 'Moderate', 'High'],
            'Your Workout': [
                1 if intensity == "Low" else 0,
                1 if intensity == "Moderate" else 0,
                1 if intensity == "High" else 0
            ]
        })
        st.bar_chart(intensity_data.set_index('Intensity'))
        
        # Heart rate zone
        st.write("Heart Rate Zones:")
        st.progress(min(heart_rate/200, 1.0))

# Additional Features
st.header("Fitness Dashboard")
tab1, tab2 = st.tabs(["Workout Trends", "Health Tips"])

with tab1:
    trend_data = pd.DataFrame({
        'Day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        'Calories': [300, 450, 400, 500, 480, 600, 350]
    })
    st.line_chart(trend_data.set_index('Day'))

with tab2:
    st.subheader("Personalized Recommendations")
    if st.session_state.prediction:
        if st.session_state.prediction > 500:
            st.success("🔥 Great intensity! Keep it up!")
            st.write("Try adding strength training 3x/week")
        else:
            st.info("💪 Good start! Try increasing workout duration by 15 minutes")
            st.write("Consider adding interval training")
    else:
        st.warning("⚠️ Click 'Calculate Calories Burned' to get recommendations")

# Deployment instructions removed from sidebar
