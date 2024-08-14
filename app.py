import streamlit as st
import pandas as pd
import pickle

model_filename = 'model_heart_disease.pkl'

with open(model_filename, 'rb') as file:
    model = pickle.load(file)

def main():
    st.title('Heart Disease Prediction')
    st.markdown('## By Saefulloh Maslul - MAI 12')
    age = st.slider('Age', 18, 100, 50)
    
    sex_options = ['Male', 'Female']
    sex = st.selectbox('Sex', sex_options)
    sex_num = 1 if sex == 'Male' else 0
    
    cp_options = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']
    cp = st.selectbox('Chest Pain Type', cp_options)
    cp_num = cp_options.index(cp)
    
    
    thalach = st.slider('Maximum Heart Rate Achieved', 70, 220, 150)
    
    exang_options = ['No', 'Yes']
    exang = st.selectbox('Exercise Induced Angina', exang_options)
    exang_num = exang_options.index(exang)
    
    oldpeak = st.slider('ST Depression Induced by Exercise Relative to Rest', 0.0, 6.2, 1.0)
    
    slope_options = ['Upsloping', 'Flat', 'Downsloping']
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', slope_options)
    slope_num = slope_options.index(slope)
    
    ca = st.slider('Number of Major Vessels Colored by Fluoroscopy', 0, 4, 1)
    
    thal_options = ['Normal', 'Fixed Defect', 'Reversible Defect']
    thal = st.selectbox('Thalassemia', thal_options)
    thal_num = thal_options.index(thal)
    
    if st.button('Predict'):
        user_input = pd.DataFrame(data={
            'age': [age],
            'sex': [sex_num],
            'cp': [cp_num],
            'thalach': [thalach],
            'exang': [exang_num],
            'oldpeak': [oldpeak],
            'slope': [slope_num],
            'ca': [ca],
            'thal': [thal_num]
        })

        prediction = model.predict(user_input)
        prediction_proba = model.predict_proba(user_input)
        
        if prediction[0] == 1:
            bg_color = 'red'
            prediction_text = 'You have heart disease'
        else:
            bg_color = 'green'
            prediction_text = 'You do not have heart disease'

        confidence = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]
        
        st.markdown(f"<p style='background-color:{bg_color}; color:white; padding:10px;'>Prediction: {prediction_text}<br>Confidence: {((confidence*10000)//1)/100}%</p>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
