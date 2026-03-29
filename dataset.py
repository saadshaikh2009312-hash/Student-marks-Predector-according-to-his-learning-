import pandas as pd
import numpy as np

np.random.seed(42)
n = 200  # Number of students

hours_study = np.random.randint(0, 20, n)
attendance = np.random.randint(50, 101, n)
assignment = np.random.randint(50, 101, n)
midterm = np.random.randint(50, 101, n)

final_grade = (
    0.3 * hours_study + 0.2 * attendance + 0.2 * assignment + 0.3 * midterm
    + np.random.normal(0, 5, n)  # Random noise
)
final_grade = np.clip(final_grade, 0, 100)  # Limit to 0-100

data = pd.DataFrame({
    "Hours_Study": hours_study,
    "Attendance": attendance,
    "Assignment_Score": assignment,
    "Midterm": midterm,
    "Final_Grade": final_grade
})

# Save to CSV
data.to_csv("student_data.csv", index=False)
print("Dataset saved as 'student_data.csv'")