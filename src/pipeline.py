import spacy
import joblib
import sys

class SkillExtractor:
    def __init__(self, ner_model_path, classifier_path):
        self.nlp = spacy.load(ner_model_path)
        self.clf = joblib.load(classifier_path)

    def predict(self, text):
        doc = self.nlp(text)
        results = []
        seen = set()  # to store normalized names

        for ent in doc.ents:
            skill_original = ent.text.strip()
            skill_norm = skill_original.lower()

            # Skip duplicates (case-insensitive)
            if skill_norm in seen:
                continue
            seen.add(skill_norm)

            label = self.clf.predict([skill_original])[0]  # "technical" or "soft"
            results.append({
                "skill": skill_original,
                "type": label
            })
        return results


if __name__ == "__main__":
    extractor = SkillExtractor("training/model-best", "models/skill_classifier.joblib")
    text = input("Enter your job description:")
    print(extractor.predict(text))