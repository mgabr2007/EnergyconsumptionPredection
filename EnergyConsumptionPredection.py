import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def main():
    st.title("Energy Consumption Prediction")
    st.write("Please upload the energy consumption data in Excel format")
    file = st.file_uploader("Upload file", type=["xlsx", "xls"])
    
    if file is not None:
        df = pd.read_excel(file)
        st.write("Here are the first few rows of the data:")
        st.dataframe(df.head())

        # Get predictor variables
        predictor_cols = st.multiselect("Select predictor variables", options=df.columns[4:13])
        num_predictors = len(predictor_cols)
        st.write("Number of predictor variables selected:", num_predictors)

        # Prepare data
        X = df[predictor_cols]
        y = df['yearly_consumption']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        st.write("Enter the predictor variable values for the prediction")
        predictor_values = []
        for col in predictor_cols:
            value = st.number_input(f"Enter {col}", value=0, step=1)
            predictor_values.append(value)
        prediction = model.predict([predictor_values])

        st.write(f"The predicted energy consumption is {prediction[0]:.2f} kWh")


if __name__ == "__main__":
    main()
