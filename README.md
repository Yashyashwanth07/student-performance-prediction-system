# 🎓 Student Performance Prediction System

## 🧾 Problem Statement
Predicting student academic performance is often challenging due to the variety of factors involved, such as academic scores, socio-economic background, behavior, and attendance. This project focuses on building an **interpretable and accessible machine learning model** that predicts student performance and provides useful feedback to educators and students.

---

## 🎯 Key Objectives
- To build a reliable system that predicts student scores based on academic, demographic, and behavioral factors.  
- To improve model accuracy using advanced ML algorithms.  
- To provide an **interactive web interface** (via Streamlit) for easy data input and real-time predictions.  
- To generate visual insights that highlight important factors influencing performance.  
- To deliver simple, actionable outputs such as star ratings and feedback suggestions.

---

## ⚙️ Methodology / Workflow
1. **Data Preprocessing**  
   - Handled missing values to maintain data consistency.  
   - Encoded categorical variables like gender, department, and grades.  
   - Split data into **training** and **testing** sets for evaluation.  

2. **Model Development**  
   - Implemented two machine learning regression models – **Random Forest** and **XGBoost**.  
   - Trained models on cleaned data and compared performance using metrics such as **R²**, **MSE**, and **MAE**.  

3. **Integration and Deployment**  
   - Saved trained models using **Joblib** for deployment.  
   - Integrated the best model (XGBoost) into a **Streamlit web app**.  
   - The app allows users to input student data, generate predictions, and visualize grades dynamically.

---

## 🧠 Tech Stack / Tools Used
| Tool / Library | Purpose |
|-----------------|----------|
| **Python** | Core programming language for implementation |
| **NumPy** | Mathematical calculations and array operations |
| **Pandas** | Data cleaning, manipulation, and analysis |
| **Matplotlib & Plotly** | Visualization and graph generation |
| **Scikit-learn** | Model training, data encoding, and evaluation |
| **XGBoost** | High-performance gradient boosting model for prediction |
| **Joblib** | Saves and loads trained ML models |
| **Streamlit** | Builds interactive web applications easily |
| **VS Code** | Development environment used for coding and debugging |

---

## 📊 Results
| Model | R² Score | MSE | MAE |
|--------|-----------|------|------|
| Random Forest | 0.954 | 2.46 | 1.31 |
| XGBoost | 0.982 | 0.96 | 0.78 |

✅ **Observation:** XGBoost produced higher accuracy and lower error metrics, making it the most suitable model for this predictive system.

---

## 🌐 Features
- Clean and interactive web-based interface for analysis.  
- Real-time performance prediction with easy visualization.  
- Model explainability through charts.  
- Personalized actionable feedback generation.

---
