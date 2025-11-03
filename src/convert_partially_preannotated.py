# Converting job_description_preannotated.jsonl to job_description_weak.csv

import pandas as pd, json

rows = []
with open("data/partially_annotated_data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        rows.append({
            "description": item["text"],
            "entities": json.dumps([
                {"start": s, "end": e, "meta_label": lbl} 
                for s, e, lbl in item["label"]
            ])
        })

pd.DataFrame(rows).to_csv("data/job_description_partially_annotated.csv", index=False)
print("✅ Saved data/job_description_partially_annotated.csv")