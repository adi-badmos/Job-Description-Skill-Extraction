import pandas as pd
import re
import json

# Load your JD CSV
df = pd.read_csv("data/job_description.csv")  # must have 'description' column

# Load seed lists
technical = [s.lower().strip() for s in pd.read_csv("data/technical_skills.csv", header=None)[0]]
soft = [s.lower().strip() for s in pd.read_csv("data/soft_skills.csv", header=None)[0]]

# Precompile regex
tech_pattern = re.compile(r'\b(' + '|'.join(map(re.escape, technical)) + r')\b', flags=re.IGNORECASE)
soft_pattern = re.compile(r'\b(' + '|'.join(map(re.escape, soft)) + r')\b', flags=re.IGNORECASE)

def find_entities(text, pattern, label):
    return [[m.start(), m.end(), label] for m in pattern.finditer(text)]

jsonl_data = []

for _, row in df.iterrows():
    text = str(row['description'])
    ents = find_entities(text, tech_pattern, "technical") + find_entities(text, soft_pattern, "soft")
    jsonl_data.append({"text": text, "label": ents})

with open("data/job_description_preannotated.jsonl", "w", encoding="utf-8") as f:
    for item in jsonl_data:
        f.write(json.dumps(item) + "\n")