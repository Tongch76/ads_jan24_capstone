import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from imblearn.over_sampling import SMOTE
import streamlit as st
import pickle

rf = pickle.load(open("heart_predict.pkl","rb"))

def main():
    st.title('Heart Disease Detection Web App')
    
    st.sidebar.title('Enter Patient Details:')
    age = st.sidebar.number_input('Age', min_value=0, max_value=120)
    gender = st.sidebar.radio('Gender', ['Male', 'Female'])
    heart_rate = st.sidebar.number_input('Heart Rate', min_value=0, max_value=300)
    sys_bp = st.sidebar.number_input('Systolic Blood Pressure', min_value=0, max_value=250)
    dia_bp = st.sidebar.number_input('Diastolic Blood Pressure', min_value=0, max_value=200)
    blood_sugar = st.sidebar.number_input('Blood Sugar', min_value=0, max_value=500)
    ck_mb = st.sidebar.number_input('CK-MB', min_value=0.0, max_value=300.0)
    troponin = st.sidebar.number_input('Troponin', min_value=0.0, max_value=11.0)

    gender_num = 1 if gender == 'Male' else 0

    user_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender_num],
        'Heart rate': [heart_rate],
        'Systolic blood pressure': [sys_bp],
        'Diastolic blood pressure': [dia_bp],
        'Blood sugar': [blood_sugar],
        'CK-MB': [ck_mb],
        'Troponin': [troponin]
    })

    prediction = rf.predict(user_data)

    st.write('### Prediction')
    if prediction[0] == 1:
        st.write(f'The model predicts: **Positive**\n')
        st.write('Please seek medical consultation **ASAP**!!!')
    else:
        st.write(f'The model predicts: **Negative**\n')
        st.write('Please do not forget to have your annual medical check-up!')

    
if __name__ == '__main__':
    main()