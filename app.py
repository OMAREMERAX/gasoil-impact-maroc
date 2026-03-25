import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# CONFIG
st.set_page_config(page_title="Gasoil Dashboard", layout="wide")

# TITLE
st.title(" Morocco Gas Oil Impact Dashboard")

# LOAD DATA
df = pd.read_csv("data/gasoil_real_maroc.csv")
df['date'] = pd.to_datetime(df['date'])

# SIDEBAR
st.sidebar.header(" Filters")

year = st.sidebar.selectbox("Select Year", df['date'].dt.year.unique())

df_filtered = df[df['date'].dt.year == year]

# METRICS
col1, col2, col3 = st.columns(3)

col1.metric("Avg Gasoil", round(df_filtered['gasoil_price'].mean(),2))
col2.metric("Avg Transport", round(df_filtered['transport_index'].mean(),2))
col3.metric("Avg Food", round(df_filtered['food_index'].mean(),2))

# CHARTS
col4, col5 = st.columns(2)

with col4:
    st.subheader(" Gasoil Evolution")
    fig1, ax1 = plt.subplots()
    ax1.plot(df_filtered['date'], df_filtered['gasoil_price'])
    plt.xticks(rotation=45)
    st.pyplot(fig1)

with col5:
    st.subheader(" Transport vs Gasoil")
    fig2, ax2 = plt.subplots()
    ax2.plot(df_filtered['date'], df_filtered['gasoil_price'], label="Gasoil")
    ax2.plot(df_filtered['date'], df_filtered['transport_index'], label="Transport")
    ax2.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig2)

# HEATMAP
st.subheader(" Correlation Matrix")
fig3, ax3 = plt.subplots()
sns.heatmap(df_filtered.corr(numeric_only=True), annot=True, ax=ax3)
st.pyplot(fig3)

# MACHINE LEARNING
st.subheader(" Transport Prediction")

X = df[['gasoil_price']]
y = df['transport_index']

model = LinearRegression()
model.fit(X, y)

gasoil_input = st.slider("Select gasoil price", 9.0, 15.0, 12.0)

prediction = model.predict([[gasoil_input]])[0]

st.success(f" Predicted Transport Index: {round(prediction,2)}")