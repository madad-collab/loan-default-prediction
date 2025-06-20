# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
@st.cache
def load_data():
    return pd.read_csv("bank_loan_default_dataset.csv")

df = load_data()

st.title("Bank Loan Default Prediction")

st.sidebar.header("Input Customer Data")

def user_input_features():
    age = st.sidebar.slider('Age', 21, 65, 35)
    income = st.sidebar.slider('Annual Income', 10000, 150000, 50000)
    loan_amount = st.sidebar.slider('Loan Amount', 5000, 50000, 20000)
    loan_term = st.sidebar.selectbox('Loan Term (months)', [12, 24, 36, 48, 60])
    credit_score = st.sidebar.slider('Credit Score', 300, 850, 650)
    employment_status = st.sidebar.selectbox('Employment Status', ['Employed', 'Self-Employed', 'Unemployed'])
    marital_status = st.sidebar.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
    dependents = st.sidebar.slider('Number of Dependents', 0, 5, 1)
    prev_defaults = st.sidebar.selectbox('Previous Defaults', [0, 1])
    debt_to_income = st.sidebar.slider('Debt-to-Income Ratio', 0.1, 0.6, 0.3)

    data = {
        'Age': age,
        'Income': income,
        'Loan_Amount': loan_amount,
        'Loan_Term_Months': loan_term,
        'Credit_Score': credit_score,
        'Employment_Status': employment_status,
        'Marital_Status': marital_status,
        'Number_of_Dependents': dependents,
        'Previous_Defaults': prev_defaults,
        'Debt_to_Income_Ratio': debt_to_income
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Preprocessing for prediction
df_full = df.drop(columns=['Customer_ID', 'Loan_Default'])
data = pd.concat([input_df, df_full], axis=0)

# Encode categorical
for col in ['Employment_Status', 'Marital_Status']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

input_processed = data[:1]

# Scale features
scaler = StandardScaler()
input_scaled = scaler.fit_transform(input_processed)

# Train model on full data
X = df.drop(['Customer_ID', 'Loan_Default'], axis=1)
y = df['Loan_Default']

for col in ['Employment_Status', 'Marital_Status']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Prediction
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

st.subheader('Input Customer Data')
st.write(input_df)

st.subheader('Prediction')
loan_status = np.array(['No Default', 'Default'])
st.write(loan_status[prediction][0])

st.subheader('Prediction Probability')
st.write(f"Probability of No Default: {prediction_proba[0][0]:.2f}")
st.write(f"Probability of Default: {prediction_proba[0][1]:.2f}")
