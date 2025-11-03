import streamlit as st
from src.predict import load_models, extract_and_label
import joblib
import os
import requests
import json

API_URL = "https://job-skill-api.onrender.com/extract"

st.set_page_config(page_title="Skill Extractor", page_icon="🧠")
st.title("Job Description Skill Extractor (spaCy NER + classifier)")

st.sidebar.header("Model paths")
ner_path = st.sidebar.text_input("spaCy NER model path", "./models/model-best")
clf_path = st.sidebar.text_input("Skill classifier path", "./models/skill_classifier.joblib")

nlp, clf = None, None
if os.path.exists(ner_path) and os.path.exists(clf_path):
    nlp, clf = load_models(ner_path, clf_path)
else:
    st.warning("Trained model(s) not found. Use the sidebar to set correct paths or train models first.")

jd = st.text_area("Paste Job Description here", height=200)

if st.button("Extract"):
    if not jd.strip():
        st.error("Please paste a job description first.")
    else:
        r = requests.post(API_URL, json={"text": jd})
        skills = r.json()

        tech = [s['span'] for s in skills if s['label']=='technical']
        soft = [s['span'] for s in skills if s['label']=='soft']

        st.subheader("Technical Skills")
        st.write(tech if tech else "—")
        st.subheader("Soft Skills")
        st.write(soft if soft else "—")
        st.subheader("All Extracted Skills")
        st.json(skills)
