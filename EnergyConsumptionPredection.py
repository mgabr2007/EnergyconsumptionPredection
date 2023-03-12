import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def main():
    st.set_page_config(page_title='Energy Consumption Prediction', layout='wide')
    st.title('Energy Consumption Prediction')

    # Allow user to upload an Excel file
    st.sidebar.title('Upload data')
    uploaded_file = st.sidebar.file_uploader('Choose a file', type=['xlsx'])

    if uploaded_file is not None:
        try:
            # Read the uploaded file into a pandas dataframe
            df = pd.read_excel(uploaded_file)

            # Display the dataframe
            st.subheader('Data')
            st.write(df)

            # Split the data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(df[['value_1', 'value_2', 'value_3']], 
                                                                df['yearly_consumption'], 
                                                                test_size=0.2, random_state=42)

            # Train the model
            lr = LinearRegression()
            lr.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = lr.predict(X_test)

            # Evaluate the model using mean squared error and R-squared
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Display model evaluation metrics
            st.subheader('Model evaluation')
            st.write(f'Mean squared error: {mse:.2f}')
            st.write(f'R-squared: {r2:.2f}')

            # Allow user to enter new data for prediction
            st.subheader('Predict')
            value_1 = st.number_input('Enter value 1', min_value=0, step=1)
            value_2 = st.number_input('Enter value 2', min_value=0, step=1)
            value_3 = st.number_input('Enter value 3', min_value=0, step=1)

            # Make prediction on new data
            y_new_pred = lr.predict([[value_1, value_2, value_3]])

            # Display predicted yearly consumption
            st.write(f'Predicted yearly consumption: {y_new_pred[0]:.2f}')

        except Exception as e:
            st.error(f'Error: {e}')


if __name__ == '__main__':
    main()
