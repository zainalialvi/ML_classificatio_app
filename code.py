import streamlit as st
import joblib

try:
    model = joblib.load('./text_classifier_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    st.title("Text Prediction App")

    input_text = st.text_input("Enter text to classify:")

    if st.button("Predict"):
        if input_text:
            input_transformed = vectorizer.transform([input_text])
            
            prediction = model.predict(input_transformed)[0]
            
            label_map = {
                1: "Positive",
                2: "Negative",
                4: "Busy",
                5: "DNC",
                8: "Not Interested",
                10: "None"
            }
    
            st.write(f"Input: {input_text}")
            st.write(f"Predicted Class: {prediction}")
            st.write(f"Label: {label_map.get(prediction, 'Unknown')}")
        else:
            st.write("Please enter some text to classify.")
except:
    st.write("some error")
