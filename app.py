import streamlit as st
from src.predict import load_models, extract_and_label
import joblib
import os

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
    if not nlp or not clf:
        st.error("Models not loaded. Train models and provide correct paths.")
    else:
        skills = extract_and_label(jd, nlp, clf)
        tech = [s['span'] for s in skills if s['label']=='technical']
        soft = [s['span'] for s in skills if s['label']=='soft']
        st.subheader("Technical Skills")
        st.write(tech if tech else "—")
        st.subheader("Soft Skills")
        st.write(soft if soft else "—")
        st.subheader("All extracted")
        st.json(skills)
