import spacy
from pathlib import Path
import joblib

def test_model(
    model_dir: str = "training/model-last",
    classifier_path: str = "models/skill_classifier.joblib"
):
    """
    Load a trained spaCy NER model and (optionally) a classifier for skill type.
    Test on a few sample job descriptions.
    """
    model_path = Path(model_dir)
    if not model_path.exists():
        print(f"❌ Model path not found: {model_path}")
        return

    print(f"✅ Loading NER model from: {model_path}")
    nlp = spacy.load(model_path)

    # Try loading classifier if available
    classifier = None
    if Path(classifier_path).exists():
        print(f"✅ Loading skill classifier from: {classifier_path}")
        classifier = joblib.load(classifier_path)
    else:
        print(f"⚠️ No classifier found at {classifier_path}. Skills will not be categorized.")

    print("\n🔍 Testing the model on sample job descriptions...\n")

    samples = [
        "Experience: 2-5 years\n\nJob Location:- Aurangabad/Pune\n\nVacancies:- 02\n\nNote: Fresher Do Not Apply\n\nJob Description\n\nLooking for experienced developers who are passionate to work with an IT / Software Development company.\n\nBasic Requirements:\nHaving prior working experience on WordPress\nShould be proficient verbally and written communication skills.\nShould be capable of writing an efficient code using best software development with good coding practices.\nAble to integrate data from various back-end services and databases.\nAble to integrate with external application ERP/CRM\nShould be capable of working on Payment gateway integration on multiple platforms\nShould have adequate knowledge of relational database systems and Object Oriented Programming.\nHands on experience required upon web applications including Security and session management.\nCapable of self- upgrading upon emerging new technologies and apply them into operations and activities.\nAble to deliver projects before deadlines.\nAble to work on multiple Frameworks such as Zend etc. would be an added advantage\nResponsibilities and Duties\nShould be able to manage on handling Multiple Projects\nManage project independently\nManage clients\nAble to handle the team\nDeliver projects before deadlines.\nRequired Experience, Skills and Qualifications\n\n\u2022 WordPress\n\u2022 Plugin-in development\n\u2022 PHP\n\u2022 HTML/HTML5\n\u2022 Javascript/jQuery\n\u2022 Bootstrap\n\u2022 MySQL\n\nQualification:\n\u2022 UG: B.Sc (CS/CSC/IT), BCA, BCS, BE, B.Tech (CS/CSE/IT)\n\u2022 M.Sc (CS/CSC/IT), MCA, MCS, ME, M.Tech (CS/CSE/IT)",
        "PYTHON/DJANGO (Developer/Lead) - Job Code(PDJ - 04)\nStrong Python experience in API development (REST/RPC).\nExperience working with API Frameworks (Django/flask).\nExperience evaluating and improving the efficiency of programs in a Linux environment.\nAbility to effectively handle multiple tasks with a high level of accuracy and attention to detail.\nGood verbal and written communication skills.\nWorking knowledge of SQL.\nJSON experience preferred.\nGood knowledge in automated unit testing using PyUnit.",
        "Data Scientist (Contractor)\n\nBangalore, IN\n\nResponsibilities\n\nWe are looking for a capable data scientist to join the Analytics team, reporting locally in India Bangalore. This person\u2019s responsibilities include research, design and development of Machine Learning and Deep Learning algorithms to tackle a variety of Fraud oriented challenges. The data scientist will work closely with software engineers and program managers to deliver end-to-end products, including: data collection in big scale and analysis, exploring different algorithmic approaches, model development, assessment and validation \u2013 all the way through production.\n\nQualifications\n\nAt least 3 years of hands-on development of complex Machine Learning models using modern frameworks and tools, ideally Python based.\nSolid understanding of statistics and applied mathematics\nCreative thinker with a proven ability to tackle open problems and apply non-trivial solutions.\nExperience in software development using Python, Java or a similar language.\nAny Graduate or M.Sc. in Computer Science, Mathematics or equivalent, preferably in Machine Learning\nAbility to write clean and concise code\nQuick learner, independent, methodical, and detail oriented.\nTeam player, positive attitude, collaborative, good communication skills.\nDedicated, makes things happen.\nFlexible, capable of making decisions in an ambiguous and changing environment.\n\nAdvantages:\n\nPrior experience as a software developer or data engineer \u2013 advantage\nExperience with Big data \u2013 advantage\nExperience with Spark \u2013 big advantage\nExperience with Deep Learning frameworks (PyTorch, TensorFlow, Keras) \u2013 advantage.\nExperience in the Telecommunication domain and/or Fraud prevention - advantage"
    ]

    for text in samples:
        doc = nlp(text)
        print("Detected Skills:")
        for ent in doc.ents:
            label = ent.label_
            skill_text = ent.text.strip()
            skill_type = None

            # Classify skill if classifier is available
            if classifier:
                try:
                    skill_type = classifier.predict([skill_text])[0]
                except Exception:
                    skill_type = "Unknown"

            # Display results neatly
            if skill_type:
                print(f" - {skill_text} ({label}, {skill_type})")
            else:
                print(f" - {skill_text} ({label})")

        print("-" * 60)


if __name__ == "__main__":
    test_model()

'''
Data Scientist (Contractor)\n\nBangalore, IN\n\nResponsibilities\n\nWe are looking for a capable data scientist to join the Analytics team, reporting locally in India Bangalore. This person\u2019s responsibilities include research, design and development of Machine Learning and Deep Learning algorithms to tackle a variety of Fraud oriented challenges. The data scientist will work closely with software engineers and program managers to deliver end-to-end products, including: data collection in big scale and analysis, exploring different algorithmic approaches, model development, assessment and validation \u2013 all the way through production.\n\nQualifications\n\nAt least 3 years of hands-on development of complex Machine Learning models using modern frameworks and tools, ideally Python based.\nSolid understanding of statistics and applied mathematics\nCreative thinker with a proven ability to tackle open problems and apply non-trivial solutions.\nExperience in software development using Python, Java or a similar language.\nAny Graduate or M.Sc. in Computer Science, Mathematics or equivalent, preferably in Machine Learning\nAbility to write clean and concise code\nQuick learner, independent, methodical, and detail oriented.\nTeam player, positive attitude, collaborative, good communication skills.\nDedicated, makes things happen.\nFlexible, capable of making decisions in an ambiguous and changing environment.\n\nAdvantages:\n\nPrior experience as a software developer or data engineer \u2013 advantage\nExperience with Big data \u2013 advantage\nExperience with Spark \u2013 big advantage\nExperience with Deep Learning frameworks (PyTorch, TensorFlow, Keras) \u2013 advantage.\nExperience in the Telecommunication domain and/or Fraud prevention - advantage
'''

'''
PYTHON/DJANGO (Developer/Lead) - Job Code(PDJ - 04)\nStrong Python experience in API development (REST/RPC).\nExperience working with API Frameworks (Django/flask).\nExperience evaluating and improving the efficiency of programs in a Linux environment.\nAbility to effectively handle multiple tasks with a high level of accuracy and attention to detail.\nGood verbal and written communication skills.\nWorking knowledge of SQL.\nJSON experience preferred.\nGood knowledge in automated unit testing using PyUnit.
'''

'''
Experience: 2-5 years\n\nJob Location:- Aurangabad/Pune\n\nVacancies:- 02\n\nNote: Fresher Do Not Apply\n\nJob Description\n\nLooking for experienced developers who are passionate to work with an IT / Software Development company.\n\nBasic Requirements:\nHaving prior working experience on WordPress\nShould be proficient verbally and written communication skills.\nShould be capable of writing an efficient code using best software development with good coding practices.\nAble to integrate data from various back-end services and databases.\nAble to integrate with external application ERP/CRM\nShould be capable of working on Payment gateway integration on multiple platforms\nShould have adequate knowledge of relational database systems and Object Oriented Programming.\nHands on experience required upon web applications including Security and session management.\nCapable of self- upgrading upon emerging new technologies and apply them into operations and activities.\nAble to deliver projects before deadlines.\nAble to work on multiple Frameworks such as Zend etc. would be an added advantage\nResponsibilities and Duties\nShould be able to manage on handling Multiple Projects\nManage project independently\nManage clients\nAble to handle the team\nDeliver projects before deadlines.\nRequired Experience, Skills and Qualifications\n\n\u2022 WordPress\n\u2022 Plugin-in development\n\u2022 PHP\n\u2022 HTML/HTML5\n\u2022 Javascript/jQuery\n\u2022 Bootstrap\n\u2022 MySQL\n\nQualification:\n\u2022 UG: B.Sc (CS/CSC/IT), BCA, BCS, BE, B.Tech (CS/CSE/IT)\n\u2022 M.Sc (CS/CSC/IT), MCA, MCS, ME, M.Tech (CS/CSE/IT)
'''