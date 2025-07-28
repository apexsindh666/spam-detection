import streamlit as st
from spam_detector import predict_message

# App title
st.set_page_config(page_title="Spam Detection", page_icon="📩")
st.title("📩 SMS Spam Classifier")
st.markdown("🚀 This app uses **Logistic Regression + TF-IDF** to detect whether a given message is **Spam** or **Ham**.")

# Text input
message = st.text_area("✍️ Enter your SMS message below:")

# Predict button
if st.button("🔍 Predict"):
    if message.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        prediction = predict_message(message)
        if prediction == "Spam":
            st.error(f"❌ Prediction: **{prediction}**")
        else:
            st.success(f"✅ Prediction: **{prediction}**")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Apex | Built using [Streamlit](https://streamlit.io)")

import joblib

# Load model and vectorizer
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

def predict_message(message):
    vec = vectorizer.transform([message])
    pred = model.predict(vec)[0]
    return "Spam" if pred == 1 else "Ham"
