import streamlit as st
import pickle

st.set_page_config(page_title="Spam Classifier")

# Load
model = pickle.load(open('models/stack.pkl', 'rb'))
tf = pickle.load(open('models/tfidf.pkl', 'rb'))

st.title("📧 Spam Email Classifier")

input_mail = st.text_input("Enter your message")

if st.button("Predict"):
    vec = tf.transform([input_mail])
    result = model.predict(vec)

    if result[0] == 0:
        st.error("🚫 Spam Mail")
    else:
        st.success("✅ Ham Mail")