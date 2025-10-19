import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

gender_encoding = {"Male": 0, "Female": 1, "Other": 2}
dept_encoding = {"Mathematics": 0, "Engineering": 1, "Business": 2, "CS": 3, "Biology": 4}
grade_encoding = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}
extracurricular_encoding = {"No": 0, "Yes": 1}
internet_encoding = {"No": 0, "Yes": 1}
parent_education_encoding = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3, "Other": 4}
income_encoding = {"Low": 0, "Medium": 1, "High": 2}

def assign_performance_rating(score):
    if score >= 85: return 5
    if 70 <= score < 85: return 4
    if 55 <= score < 70: return 3
    if 40 <= score < 55: return 2
    return 1

def provide_advice(rating, features):
    advice = []
    if rating < 3:
        advice.append("Improve participation in class activities.")
        advice.append("Increase study hours to enhance learning outcomes.")
    if features['attendance'] < 75:
        advice.append("Try to improve attendance to avoid missing critical content.")
    if features['stress'] > 7:
        advice.append("Consider stress management strategies to improve focus.")
    if features['sleep'] < 6:
        advice.append("Aim for at least 6-8 hours of sleep per night for optimal performance.")
    if features['extracurricular'] == 0:
        advice.append("Engage in extracurricular activities to develop holistic skills.")
    if not advice:
        advice.append("Maintain your excellent habits!")
    return advice

st.set_page_config(layout="wide")
st.title("Student Performance Analyzer")

with st.sidebar:
    selected_model = st.radio("Choose Regression Model", ["Random Forest", "XGBoost"])
    st.markdown("Input student details below:")

with st.container():
    group_a, group_b, group_c = st.columns(3)
    with group_a:
        chosen_gender = st.radio("Gender", list(gender_encoding.keys()))
        input_age = st.number_input("Age (years)", min_value=10, max_value=100, value=20)
        chosen_department = st.selectbox("Department", list(dept_encoding.keys()))
        input_attendance = st.slider("Attendance (%)", 0.0, 100.0, 85.0)
        extracurricular_flag = st.radio("Extracurricular Activities", list(extracurricular_encoding.keys()))
        internet_flag = st.radio("Internet Access at Home", list(internet_encoding.keys()))
    with group_b:
        input_midterm = st.slider("Midterm Score", 0.0, 100.0, 75.0)
        input_final = st.slider("Final Score", 0.0, 100.0, 80.0)
        avg_assignments = st.slider("Assignments Average", 0.0, 100.0, 70.0)
        avg_quizzes = st.slider("Quizzes Average", 0.0, 100.0, 80.0)
        input_participation = st.slider("Participation", 0.0, 100.0, 78.0)
        input_projects = st.slider("Projects Total", 0.0, 100.0, 90.0)
    with group_c:
        input_grade = st.selectbox("Grade", list(grade_encoding.keys()))
        time_studying = st.slider("Weekly Study Hours", 0.0, 100.0, 10.0)
        parent_education = st.selectbox("Parent's Education", list(parent_education_encoding.keys()))
        family_income = st.selectbox("Family Income", list(income_encoding.keys()))
        stress_index = st.slider("Self-evaluated Stress (1-10)", 1.0, 10.0, 5.0)
        sleep_average = st.slider("Average Sleep (hours/night)", 0.0, 24.0, 8.0)

student_features = np.array([[
    gender_encoding[chosen_gender], input_age, dept_encoding[chosen_department], input_attendance,
    input_midterm, input_final, avg_assignments, avg_quizzes, input_participation,
    input_projects, grade_encoding[input_grade], time_studying, extracurricular_encoding[extracurricular_flag],
    internet_encoding[internet_flag], parent_education_encoding[parent_education],
    income_encoding[family_income], stress_index, sleep_average
]])

rf_model = joblib.load('randomforest.pkl')
xgb_model = joblib.load('xgboost_model.pkl')

if st.button("Get Performance Report"):
    if selected_model == "Random Forest":
        final_score = rf_model.predict(student_features)[0]
    else:
        final_score = xgb_model.predict(student_features)[0]

    rating = assign_performance_rating(final_score)

    st.markdown("---")
    st.header("Personalized Student Performance Report")

    features_dict = {
        'attendance': input_attendance,
        'extracurricular': extracurricular_encoding[extracurricular_flag],
        'stress': stress_index,
        'sleep': sleep_average
    }

    col1, col2 = st.columns([3, 2])
    with col1:
        st.success(f"Score Prediction: {final_score:.2f}")
        st.markdown(f"### Performance Star Rating: {'★' * rating} ({rating}/5)")

        advice_list = provide_advice(rating, features_dict)
        st.markdown("### Suggestions to Improve Performance:")
        for advice in advice_list:
            st.write(f"- {advice}")

        features_labels = [
            "Gender", "Age", "Department", "Attendance", "Midterm", "Final", "Assignments",
            "Quizzes", "Participation", "Projects", "Grade", "Study Hours", "Extracurricular",
            "Internet", "Parent Edu", "Income", "Stress", "Sleep"
        ]
        feature_values = student_features[0]
        fig, ax = plt.subplots(figsize=(15, 4))
        ax.bar(features_labels, feature_values, color="#5b9bd5")
        ax.set_xticklabels(features_labels, rotation=35, ha='right', fontsize=10)
        ax.set_title("Profile Attributes Input Summary")
        st.pyplot(fig)
