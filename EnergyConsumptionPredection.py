import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Disable PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Read data
file = st.file_uploader("Upload file", type=["xlsx"])
if file:
    df = pd.read_excel(file)

    # Select predictor variables
    predictor_cols = st.multiselect("Select predictor variables", list(df.columns))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(df[predictor_cols], df['yearly_consumption'], test_size=0.2)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on input
    input_data = []
    for col in predictor_cols:
        value = st.number_input(f"Enter {col}")
        input_data.append(value)
    prediction = model.predict([input_data])

    # Display prediction
    st.write("Predicted yearly consumption:", prediction[0])

    # Plot relationship between predictor variables and consumption
    fig, ax = plt.subplots()
    sns.pairplot(df, x_vars=predictor_cols, y_vars='yearly_consumption', height=5, aspect=0.7, kind='reg')
    st.pyplot(fig)
