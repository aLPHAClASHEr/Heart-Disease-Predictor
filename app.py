import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('heart_disease_model.sav', 'rb'))


# Creating a function for the prediction
def heart_disease_predictor(input_data):
    new_array = np.asarray(input_data)

    # Reshaping the numpy array as we are predicting for only one instance
    input_data_reshaped = new_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    # print(prediction)

    if prediction[0] == 0:
        return'The person does not have a heart Disease'
    else:
        return'The person has Heart Disease'


def main():
    # Giving a title for the Web App
    st.title('Heart Disease Predictor')

    # Getting the input data from the user
    #age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal

    age = st.text_input('Enter the Age:')
    sex = st.text_input('Enter the Gender of the person:')
    cp = st.text_input('Enter the Cardiac Pulse:')
    trestbps = st.text_input('Enter the TRESTBPS:')
    chol = st.text_input('Enter the Cholesterol level:')
    fbs = st.text_input('Enter the FBS:')
    restecg = st.text_input('Enter the RESTECG:')
    thalach = st.text_input('Enter the THALACH:')
    exang = st.text_input('Enter the EXANG:')
    oldpeak = st.text_input('Enter the OLDPEAK:')
    slope = st.text_input('Enter the SLOPE:')
    ca = st.text_input('Enter the CA:')
    thal = st.text_input('Enter the THAl:')

    # Code for prediction
    diagonosis = ''

    # Creating a button for the user to predict the heart disease
    if st.button('Predict Heart Disease'):
        diagonosis = heart_disease_predictor(
            [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])

    st.success(diagonosis)


if __name__ == '__main__':
    main()


# This code cannot run directly
# This code has to be run through the terminal
# Using this command
# streamlt run "File_location\file_name.py"    Then press 'Enter'
# This directs to a local host webpage created by streamlit
# To stop the web app in terminal press Ctrl + C
