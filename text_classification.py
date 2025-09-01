import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("best_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit UI
st.set_page_config(page_title="Report Category Classifier", page_icon="📊")
st.title("📊 Report Category Classification App")
st.write("Enter your query below and the model will predict the most relevant report category:")

# User input
user_query = st.text_input("Enter your query:")

if st.button("Predict"):
    if user_query.strip() == "":
        st.warning("⚠️ Please enter a query!")
    else:
        # Transform query and predict
        query_tfidf = vectorizer.transform([user_query])
        prediction = model.predict(query_tfidf)[0]

        st.success(f"✅ Predicted Category: **{prediction}**")


