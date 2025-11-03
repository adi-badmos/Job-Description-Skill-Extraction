import spacy
import joblib
from pathlib import Path

def load_models(ner_model_path: str = None, classifier_path: str = None):
    # Load spaCy NER model
    if ner_model_path:
        nlp = spacy.load(ner_model_path)
    else:
        nlp = spacy.load("en_core_web_sm")

    # Load classifier
    clf = joblib.load(classifier_path) if classifier_path and Path(classifier_path).exists() else None
    return nlp, clf

def extract_and_label(text: str, nlp, clf):
    doc = nlp(text)

    STOP_WORD_SKILLS = {
        "ability", "experience", "knowledge", "skills", "attitude",
        "team", "teams", "environment", "development"
    }

    seen = set()
    skills = []

    for ent in doc.ents:
        if ent.label_.upper() != 'SKILL':
            continue

        span_text = ent.text.strip()
        norm = span_text.lower()

        # Skip generic bad words
        if norm in STOP_WORD_SKILLS:
            continue

        # Skip duplicates (case-insensitive)
        if norm in seen:
            continue
        seen.add(norm)

        # Classifier label
        label = clf.predict([span_text])[0] if clf else None

        skills.append({
            'span': span_text,
            'start': ent.start_char,
            'end': ent.end_char,
            'label': label
        })

    return skills


if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument("--ner", default="/training/model-best", help="trained spaCy model dir")
    p.add_argument("--clf", required=True, help="skill classifier joblib")
    p.add_argument("--text", required=True)
    args = p.parse_args()
    nlp, clf = load_models(args.ner, args.clf)
    res = extract_and_label(args.text, nlp, clf)
    print(json.dumps(res, indent=2))