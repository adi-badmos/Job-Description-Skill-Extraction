from fastapi import File, UploadFile
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.predict import load_models, extract_and_label

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

nlp, clf = load_models("./training/model-best", "./models/skill_classifier.joblib")

@app.post("/extract")
def extract_api(data: dict):
    jd = data.get("text", "")
    return extract_and_label(jd, nlp, clf)

@app.post("/extract_file")
async def extract_from_file(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")

    # Try reading as CSV first
    try:
        df = pd.read_csv(pd.io.common.StringIO(text))
        # If CSV, combine all columns into one text blob
        text = " ".join(df.astype(str).values.flatten())
    except Exception:
        # If not CSV, treat entire content as raw text
        pass

    return extract_and_label(text, nlp, clf)