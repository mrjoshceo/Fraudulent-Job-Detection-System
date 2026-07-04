
# Import required libraries
import streamlit as st
import joblib
import re
import string

# Load the trained model
model = joblib.load("svm_model.pkl")

# Load the TF-IDF vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Function to clean text
def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# -------------------------------
# Streamlit User Interface
# -------------------------------

st.title("Fraudulent Job Detection System")

st.write(
    "Enter a job description below to determine whether it is Genuine or Fraudulent."
)

job_text = st.text_area(
    "Job Description",
    height=250
)

# Predict button
if st.button("Predict"):

    # Check that the user entered text
    if job_text.strip() == "":
        st.warning("Please enter a job description.")

    else:

        # Clean the input text
        cleaned_text = clean_text(job_text)

        # Convert text to TF-IDF features
        text_vector = vectorizer.transform([cleaned_text])

       # Make prediction
prediction = model.predict(text_vector)[0]

# Generate confidence score
confidence = model.predict_proba(text_vector).max() * 100

# Display the result
if prediction == 0:
    st.success("Prediction: Genuine Job")
else:
    st.error("Prediction: Fraudulent Job")

# Display confidence score
st.info(f"Confidence Score: {confidence:.2f}%")