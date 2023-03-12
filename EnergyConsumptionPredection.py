import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    st.title("Energy Consumption Prediction App")

    # File upload
    st.header("Upload your dataset")
    file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

    if file is not None:
        try:
            # Read dataset
            df = pd.read_excel(file)

            # Column selection
            st.header("Select the predictor variables")
            predictor_cols = st.multiselect("Select columns", options=list(df.columns))

            # Number of predictor variables
            num_predictors = len(predictor_cols)
            st.write("Number of predictor variables:", num_predictors)

            # Target column selection
            st.header("Select the target variable")
            target_col = st.selectbox("Select column", options=list(df.columns))

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(df[predictor_cols], df[target_col], test_size=0.2)

            # Model training
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Model evaluation
            score = model.score(X_test, y_test)
            st.write("Model score:", score)

            # Prediction input
            st.header("Enter predictor variable values for prediction")
            input_dict = {}
            for col in predictor_cols:
                val = st.number_input(f"Enter {col} value")
                input_dict[col] = [val]
            input_df = pd.DataFrame.from_dict(input_dict)

            # Prediction
            prediction = model.predict(input_df)
            st.write("Predicted value:", prediction[0])

            # Graph
            st.header("Graph of the data")
            sns.scatterplot(x=predictor_cols[0], y=target_col, data=df)
            plt.xlabel(predictor_cols[0])
            plt.ylabel(target_col)
            st.pyplot()

        except Exception as e:
            st.write("An error occurred:", e)


if __name__ == "__main__":
    main()
