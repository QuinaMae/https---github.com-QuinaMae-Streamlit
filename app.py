import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model and scaler
model = joblib.load('models\model.pkl')
scaler = joblib.load('models\scaler.pkl')
label_encoders = joblib.load('models/label_encoders.pkl')

# Function to preprocess new data point
def preprocess_data(new_data):
    # Encode categorical features
    new_data_encoded = new_data.copy()
    categorical_features = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
    for feature in categorical_features:
        label_encoder = label_encoders[feature]
        new_data_encoded[feature] = label_encoder.transform(new_data_encoded[feature].astype(str))

    # Preprocess the new data point
    new_data_scaled = scaler.transform(new_data_encoded)
    return new_data_scaled

# Function to predict attrition
def predict_attrition(data):
    # Preprocess the data
    data_scaled = preprocess_data(data)

    # Predict attrition
    attrition_result = model.predict(data_scaled)

    # Return the result
    return attrition_result

# Main Streamlit app
def main():
    # Set app title
    st.title("StayOnTrack: Navigating Employee Attrition with Precision")
    st.markdown('A simple application for predicting attrition')

    # Add user input section
    st.subheader("Employee Profile")
    age = st.number_input("Age", min_value=18, max_value=100)
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"])
    education = st.selectbox("Education (1 'Below College' 2 'College' 3 'Bachelor' 4 'Master' 5 'Doctor')", [1, 2, 3, 4, 5])
    num_companies_worked = st.number_input("Number of Companies Worked")
    total_working_years = st.number_input("Total Working Years")
    
    st.subheader("Current Job")
    distance_from_home = st.number_input("Distance From Home")
    stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3])
    years_at_company = st.number_input("Years at Company")
    years_in_current_role = st.number_input("Years in Current Role")
    years_since_last_promotion = st.number_input("Years Since Last Promotion")
    years_with_curr_manager = st.number_input("Years with Current Manager")
    department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    job_role = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])
    over_time = st.selectbox("Over Time", ["Yes", "No"])
    business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])

    st.subheader("Salary")
    daily_rate = st.number_input("Daily Rate")
    hourly_rate = st.number_input("Hourly Rate")
    monthly_income = st.number_input("Monthly Income")
    monthly_rate = st.number_input("Monthly Rate")
    percent_salary_hike = st.number_input("Percent Salary Hike")
    training_times_last_year = st.number_input("Training Times Last Year")

    st.subheader("Job Wellness")
    environment_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4, 5])
    job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4])
    job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
    job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4, 5])
    performance_rating = st.selectbox("Performance Rating", [1, 2, 3, 4, 5])
    relationship_satisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4, 5])
    work_life_balance = st.selectbox("Work Life Balance", [1, 2, 3, 4, 5])
    
    
    #Create Predict btn
    button_clicked = st.button("Predict", help="Click to predict attrition")
    if button_clicked:
    # Create a new data point based on user inputs
        new_data = pd.DataFrame({
            'Age': [age],
            'DailyRate': [daily_rate],
            'DistanceFromHome': [distance_from_home],
            'Education': [education],
            'EnvironmentSatisfaction': [environment_satisfaction],
            'HourlyRate': [hourly_rate],
            'JobInvolvement': [job_involvement],
            'JobLevel': [job_level],
            'JobSatisfaction': [job_satisfaction],
            'MonthlyIncome': [monthly_income],
            'MonthlyRate': [monthly_rate],
            'NumCompaniesWorked': [num_companies_worked],
            'PercentSalaryHike': [percent_salary_hike],
            'PerformanceRating': [performance_rating],
            'RelationshipSatisfaction': [relationship_satisfaction],
            'StockOptionLevel': [stock_option_level],
            'TotalWorkingYears': [total_working_years],
            'TrainingTimesLastYear': [training_times_last_year],
            'WorkLifeBalance': [work_life_balance],
            'YearsAtCompany': [years_at_company],
            'YearsInCurrentRole': [years_in_current_role],
            'YearsSinceLastPromotion': [years_since_last_promotion],
            'YearsWithCurrManager': [years_with_curr_manager],
            'BusinessTravel': [business_travel],
            'Department': [department],
            'EducationField': [education_field],
            'Gender': [gender],
            'JobRole': [job_role],
            'MaritalStatus': [marital_status],
            'OverTime': [over_time]
            
        })

        # Predict attrition for the new data point
        attrition_result = predict_attrition(new_data)

            # Display the result
        st.subheader("Result")
        if attrition_result == 1:
            # st.write("The employee is most likely to churn or quit")
            st.markdown(
                """
                <div style="background-color: #FFCCCC; padding: 10px; border-radius: 5px;">
                    <p style="font-weight: bold; color: #FF0000;">The employee is most likely to churn or quit</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            # st.write("The employee is most likely far from resigning or quitting")
            st.markdown(
                """
                <div style="background-color: #CCFFCC; padding: 10px; border-radius: 5px;">
                    <p style="font-weight: bold; color: #008000;">The employee is most likely far from resigning or quitting</p>
                </div>
                """,
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()
