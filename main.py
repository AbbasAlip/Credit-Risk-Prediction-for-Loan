import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

# Load the data
df = pd.read_csv("german_credit_data.csv")

# Drop unnamed index column if present
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Preprocess the data
def preprocess_data(df):
    df = df.copy()

    # Encode categorical columns
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Feature scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_imputed)

    return scaled_data, label_encoders, scaler, imputer, df.columns

X, label_encoders, scaler, imputer, feature_columns = preprocess_data(df)

# Train KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Streamlit UI
st.title("Credit Profile Clustering App")
st.write("This app uses KMeans clustering to group customers based on financial and demographic attributes.")

input_data = {}
for col in feature_columns:
    if df[col].dtype == 'object':
        input_data[col] = st.selectbox(col, df[col].unique())
    else:
        input_data[col] = st.number_input(col, value=float(df[col].mean()))

if st.button("Predict Cluster"):
    input_df = pd.DataFrame([input_data])

    # Apply label encoding
    for col, le in label_encoders.items():
        input_df[col] = le.transform(input_df[col])

    # Impute and scale
    input_df_imputed = pd.DataFrame(imputer.transform(input_df), columns=input_df.columns)
    input_scaled = scaler.transform(input_df_imputed)

    cluster = kmeans.predict(input_scaled)[0]

    st.subheader("Prediction Result")
    st.write(f"The input data belongs to **Cluster {cluster}**.")
