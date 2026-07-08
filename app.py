
# Import required libraries
import streamlit as st
import joblib
import re
import string

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Fraudulent Job Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🛡️ Fraud Detection System")

st.sidebar.markdown("""
### About

This application uses a Machine Learning Support Vector Machine (SVM) model to classify job advertisements as either:

- ✅ Genuine
- 🚨 Fraudulent

The model was trained using TF-IDF text features extracted from job descriptions.

---

### Machine Learning Model

- Algorithm: Support Vector Machine (SVM)
- Feature Extraction: TF-IDF
- Language: Python
- Framework: Streamlit

---

### Developed For

BSc (Hons) Computing with Cyber Security Technology

Northumbria University
""")

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

st.title("🛡️ Fraudulent Job Detection System")

st.markdown("""
### Machine Learning-Based Fraud Detection

This application analyses job advertisement text using a trained **Support Vector Machine (SVM)** model.

Paste a job description below and click **Predict** to determine whether the advertisement is likely to be **Genuine** or **Fraudulent**.

---
""")

job_text = st.text_area(
    "📄 Enter Job Description",
    height=300,
    placeholder="""Example:

Software Engineer

Responsibilities:
- Develop Python applications
- Collaborate with the engineering team
- Maintain software systems

Requirements:
- Bachelor's degree
- 2+ years experience
- Strong communication skills
"""
)

if st.button("Predict"):

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

        # ----------------------------------
# Prediction Dashboard
# ----------------------------------

st.markdown("---")
st.header("📊 Prediction Dashboard")

# Determine labels
prediction_label = "Genuine" if prediction == 0 else "Fraudulent"
status = "Safe" if prediction == 0 else "High Risk"

# Create three columns
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Prediction",
        value=prediction_label
    )

with col2:
    st.metric(
        label="Confidence",
        value=f"{confidence:.2f}%"
    )

with col3:
    st.metric(
        label="Algorithm",
        value="SVM"
    )

# Second Row
col4, col5, col6 = st.columns(3)

with col4:
    st.metric(
        label="Feature Extraction",
        value="TF-IDF"
    )

with col5:
    st.metric(
        label="Dataset",
        value="Fake Jobs"
    )

with col6:
    st.metric(
        label="Status",
        value=status
    )

st.write("")

# Progress Bar
st.progress(min(confidence / 100, 1.0))

# Colour result

if prediction == 0:
    st.success("✅ Genuine Job Advertisement")
else:
    st.error("🚨 Fraudulent Job Advertisement")

# Confidence message

if confidence >= 90:
    st.info("🟢 High Confidence Prediction")
elif confidence >= 70:
    st.warning("🟡 Moderate Confidence Prediction")
else:
    st.error("🔴 Low Confidence Prediction")