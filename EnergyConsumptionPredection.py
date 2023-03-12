import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Energy Consumption Prediction")

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.write(df)

    col_list = df.columns.tolist()
    col_selected = st.multiselect('Select the predictor variables:', col_list)
    st.write('You selected:', len(col_selected), 'predictor variables')

    X = df[col_selected]
    y = df['yearly_consumption']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    lm = LinearRegression()
    lm.fit(X_train, y_train)

    predictions = lm.predict(X_test)

    fig, ax = plt.subplots()
    ax.scatter(y_test, predictions)
    ax.plot(y_test, y_test, 'r')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    st.pyplot(fig)

    st.write('R^2 Score:', lm.score(X_test, y_test))
