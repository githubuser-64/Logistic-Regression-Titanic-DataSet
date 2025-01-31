import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm

st.title("🎈 My Logistic Regression app")
st.write(
    "Your task includes creating a Streamlit app in Python that involves loading your trained model and setting up user inputs for predictions.  [docs.streamlit.io](https://docs.streamlit.io/)."
)

def add_constant_column(df):
  """Adds a 'const' column with value 1 as the first column of a DataFrame.

  Args:
      df (pd.DataFrame): The input DataFrame.

  Returns:
      pd.DataFrame: The DataFrame with the added 'const' column.
  """
  df.insert(0, 'const', 1)
  return df

# Function to calculate predicted probability
def predict_survival_probability(input_data, model_params):
    """
    Calculates the predicted survival probability using the model parameters.

    Args:
        input_data (dict): User inputs as a dictionary.
        model_params (pd.Series): Model's coefficients.

    Returns:
        float: Predicted survival probability.
    """
    # Create dataframe from input and add a constant for the bias
    input_df = pd.DataFrame([input_data])
    input_df = add_constant_column(input_df)
    input_values = input_df[model_params.index].values

    # Calculate the prediction
    log_odds = np.dot(input_values, model_params.values)
    probability = 1 / (1 + np.exp(-log_odds))
    return probability

# Streamlit app
st.title("Titanic Survival Prediction App")

# Model parameters (replace with your actual parameters from the results of the model object)
model_params = pd.Series({
    "const":      1.1042,
    "Pclass":    -0.6105,
    "Sex":      23.4972,
    "Age":       -0.0255,
    "SibSp":     -0.5764,
    "Parch":      -0.4034,
    "Fare":        0.0056,
    "Title_Dr":   -0.0285,
    "Title_Master":2.3935,
    "Title_Miss":-21.6615,
    "Title_Mr":  -1.0110,
    "Title_Mrs":-20.3297,
    "Title_Ms":   9.9564,
    "Title_Rev":-22.2649,
    "First_Letter_Cabin_B":     0.9785,
    "First_Letter_Cabin_C":     0.4258,
    "First_Letter_Cabin_D":     1.5744,
    "First_Letter_Cabin_E":     2.0160,
    "First_Letter_Cabin_F":     1.0553,
    "First_Letter_Cabin_G":    -1.5669,
    "First_Letter_Cabin_χ":     0.1763,
})


# Input widgets (using values in params)
pclass = st.selectbox("Pclass", options=[1, 2, 3])
sex = st.selectbox("Sex", options = [0,1]) # male = 0, female = 1, which is what you used to encode
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.number_input("SibSp", min_value=0, max_value=10, value=0)
parch = st.number_input("Parch", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0, value=20)
title_options = ['Dr','Master','Miss','Mr','Mrs','Ms','Rev']
title = st.selectbox("Title", options=title_options)

first_letter_cabin_options = ['B','C','D','E','F','G','χ']
first_letter_cabin = st.selectbox("First letter of Cabin", options = first_letter_cabin_options)

# Create input dictionary
input_data = {
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Title_Dr": 1 if title == "Dr" else 0,
    "Title_Master": 1 if title == "Master" else 0,
    "Title_Miss": 1 if title == "Miss" else 0,
    "Title_Mr": 1 if title == "Mr" else 0,
    "Title_Mrs": 1 if title == "Mrs" else 0,
     "Title_Ms": 1 if title == "Ms" else 0,
    "Title_Rev": 1 if title == "Rev" else 0,
    "First_Letter_Cabin_B":1 if first_letter_cabin == "B" else 0,
    "First_Letter_Cabin_C": 1 if first_letter_cabin == "C" else 0,
    "First_Letter_Cabin_D": 1 if first_letter_cabin == "D" else 0,
    "First_Letter_Cabin_E": 1 if first_letter_cabin == "E" else 0,
    "First_Letter_Cabin_F": 1 if first_letter_cabin == "F" else 0,
    "First_Letter_Cabin_G": 1 if first_letter_cabin == "G" else 0,
    "First_Letter_Cabin_χ": 1 if first_letter_cabin == "χ" else 0,

}

# Prediction button
if st.button("Predict Survival"):
    probability = predict_survival_probability(input_data, model_params)
    st.write(f"Predicted Survival Probability: {probability.round(3)}")
    st.write(f"The person will probably {'survive.' if probability > 0.5 else 'not survive.'}")
