import streamlit as st
import joblib
import re

# Set up the page layout
st.set_page_config(page_title="Insurance Review Analyzer", layout="centered")

st.title("Insurance Review Sentiment Analyzer")
st.write("Type a customer review in English below to predict its star rating!")

# Load the saved models
@st.cache_resource
def load_models():
    vec = joblib.load('tfidf_vectorizer.pkl')
    mod = joblib.load('logistic_model.pkl')
    return vec, mod

try:
    vectorizer, model = load_models()
except Exception as e:
    st.error("Model files not found! Please make sure you ran the export cell in your notebook.")

# Create the text input box
user_input = st.text_area("Customer Review:", height=150, placeholder="The customer service was fantastic and the price is great!")

if st.button("Predict Star Rating"):
    if user_input.strip() == "":
        st.warning("Please enter a review first!")
    else:
        # Clean the text slightly
        clean_text = re.sub(r'[^\w\s]', '', user_input.lower())
        
        # Vectorize the input
        vectorized_text = vectorizer.transform([clean_text])
        
        # Make the prediction
        prediction = model.predict(vectorized_text)[0]
        
        # Display the result beautifully
        st.subheader("Prediction Result")
        st.write(f"This review looks like a **{int(prediction)} Star** rating!")
        
        # Display visual stars
        st.write("⭐" * int(prediction))