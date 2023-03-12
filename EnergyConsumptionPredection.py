import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.title('Energy Consumption Prediction')

# Allow user to upload an Excel file
file = st.file_uploader('Upload Excel file', type=['xls', 'xlsx'])

if file is not None:
    # Read the Excel file into a pandas dataframe
    df = pd.read_excel(file)

    # Drop any rows with missing values
    df.dropna(inplace=True)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df[['value_1', 'value_2', 'value_3']], df['yearly_consumption'], test_size=0.2)

    # Train a linear regression model on the training data
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Calculate the root mean squared error (RMSE) of the predictions
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    # Display the RMSE to the user
    st.write(f'Root Mean Squared Error: {rmse:.2f}')
else:
    st.write('Please upload an Excel file.')
