import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="Student Performance Predictor", layout="centered")
st.title("🎓 Student Performance Predictor")

# -------------------------------
# 1. Load trained model
# -------------------------------
model = joblib.load("student_model.pkl")

# -------------------------------
# 2. Input from user
# -------------------------------
st.subheader("Enter Your Details")

hours = st.number_input("Hours Studied per Week", 0, 100, 10)
attendance = st.number_input("Attendance Percentage", 0, 100, 80)
assignment = st.number_input("Average Assignment Score", 0, 100, 75)
midterm = st.number_input("Midterm Exam Score", 0, 100, 70)

if st.button("Predict Final Grade"):
    prediction = model.predict([[hours, attendance, assignment, midterm]])
    st.success(f"Predicted Final Grade: {prediction[0]:.2f} / 100")

    # Recommendation
    if prediction[0] >= 85:
        st.info("Excellent! Keep up the good work! 🎉")
    elif prediction[0] >= 70:
        st.warning("Good! You can still improve by studying a bit more. 📚")
    else:
        st.error("Needs Improvement. Consider increasing study hours and focus on assignments. ⚠️")

# -------------------------------
# 3. Generate mock data for visualization
# -------------------------------
@st.cache_data
def generate_data(n=500):
    np.random.seed(42)
    hours_study = np.random.randint(0, 20, n)
    attendance = np.random.randint(50, 101, n)
    assignment = np.random.randint(50, 101, n)
    midterm = np.random.randint(50, 101, n)
    final_grade = (
        0.3 * hours_study + 0.2 * attendance + 0.2 * assignment + 0.3 * midterm
        + np.random.normal(0, 5, n)
    )
    final_grade = np.clip(final_grade, 0, 100)
    data = pd.DataFrame({
        "Hours_Study": hours_study,
        "Attendance": attendance,
        "Assignment_Score": assignment,
        "Midterm": midterm,
        "Final_Grade": final_grade
    })
    return data

data = generate_data()

# -------------------------------
# 4. Show correlation heatmap
# -------------------------------
st.subheader("Feature Correlation with Final Grade")
fig, ax = plt.subplots()
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# -------------------------------
# 5. CSV upload for batch predictions
# -------------------------------
st.subheader("Batch Prediction via CSV Upload")
uploaded_file = st.file_uploader("Upload CSV with columns: Hours_Study, Attendance, Assignment_Score, Midterm", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if all(col in df.columns for col in ["Hours_Study", "Attendance", "Assignment_Score", "Midterm"]):
        df["Predicted_Grade"] = model.predict(df[["Hours_Study", "Attendance", "Assignment_Score", "Midterm"]])
        st.success("Batch Prediction Complete!")
        st.dataframe(df)
        df.to_csv("predicted_grades.csv", index=False)
        st.download_button("Download Predictions CSV", "predicted_grades.csv")
    else:
        st.error("CSV missing required columns.")