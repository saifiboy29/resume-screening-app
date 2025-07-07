import streamlit as st
import re
import nltk
import pickle
from PyPDF2 import PdfReader  # For reading PDFs

# NLTK downloads
nltk.download("punkt")
nltk.download("stopwords")

# Load saved models
tfidv = pickle.load(open("tfidv.pkl", "rb"))
knc = pickle.load(open("knc.pkl", "rb"))

# Category mapping
category_count = {
    15: "Java Developer", 23: "Testing", 8:  "DevOps Engineer", 20: "Python Developer",
    24: "Web Designing", 12: "HR", 13: "Hadoop", 22: "Sales", 6:  "Data Science",
    16: "Mechanical Engineer", 10: "ETL Developer", 3:  "Blockchain", 18: "Operations Manager",
    1:  "Arts", 7:  "Database", 14: "Health and fitness", 19: "PMO", 11: "Electrical Engineering",
    4:  "Business Analyst", 9:  "DotNet Developer", 2:  "Automation Testing",
    17: "Network Security Engineer", 5:  "Civil Engineer", 21: "SAP Developer", 0:  "Advocate"
}

# Resume cleaning function
def clean_resume(txt):
    txt = re.sub(r'http\S+', '', txt)
    txt = re.sub(r'@\w+', '', txt)
    txt = re.sub(r'#\w+', '', txt)
    txt = re.sub(r'[^a-zA-Z ]', ' ', txt)
    txt = txt.lower()
    return txt

# File text extraction
def extract_text(file):
    if file.type == "application/pdf":
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    elif file.type == "text/plain":
        return str(file.read(), "utf-8")
    else:
        return ""

# Streamlit UI
st.title("📄 Resume Screening App")
st.write("Upload a resume file (PDF or TXT) to predict the job role.")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "txt"])

if uploaded_file is not None:
    resume_text = extract_text(uploaded_file)

    if st.button("Predict Job Role"):
        if resume_text.strip() == "":
            st.warning("The file appears to be empty or unreadable.")
        else:
            cleaned = clean_resume(resume_text)
            vectorized = tfidv.transform([cleaned])
            prediction = knc.predict(vectorized)[0]
            role = category_count.get(prediction, "Unknown")
            st.success(f"🧠 Predicted Job Role: **{role}**")
