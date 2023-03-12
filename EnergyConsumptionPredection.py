import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load data from file uploaded by user
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    # Create a list of all columns except the target variable and the 'id' column
    predictor_columns = [col for col in df.columns if col not in ['id', 'yearly_consumption']]
    
    # Add a dropdown list to select predictor variables
    selected_columns = st.multiselect('Select predictor variables', predictor_columns)

    if len(selected_columns) > 0:
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(df[selected_columns], df['yearly_consumption'], test_size=0.2)

        # Train a linear regression model on the training set
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict the yearly consumption for the user-input values
        st.write('Enter predictor values:')
        input_values = []
        for col in selected_columns:
            val = st.number_input(col)
            input_values.append(val)
        predicted_consumption = model.predict([input_values])

        # Display the predicted yearly consumption
        st.write('Predicted yearly consumption:', predicted_consumption[0])
