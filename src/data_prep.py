import argparse
import pandas as pd
import json
import re
from pathlib import Path
from spacy.tokens import DocBin
import spacy
from spacy.util import filter_spans
from tqdm import tqdm


nlp = spacy.blank("en")


def read_seed_list(path: Path):
    if not path.exists():
        return set()
    return set(pd.read_csv(path, header=None)[0].str.lower().str.strip().tolist())


def find_span_offsets(text, phrase):
    spans = []
    for match in re.finditer(re.escape(phrase), text, flags=re.IGNORECASE):
        spans.append((match.start(), match.end()))
    return spans


def parse_entities_field(x):
    try:
        return json.loads(x)
    except Exception:
        return None


def guess_label(skill_text, technical_seeds, soft_seeds):
    t = skill_text.lower().strip()
    if t in technical_seeds:
        return 'technical'
    if t in soft_seeds:
        return 'soft'
    return 'unknown'


def make_docbin(examples, out_path):
    db = DocBin(store_user_data=True)
    for doc, ents in examples:
        spans = []
        for (start, end, label) in ents:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is not None:
                spans.append(span)
        spans = filter_spans(spans)
        doc.ents = spans
        db.add(doc)
    db.to_disk(out_path)
    print(f"✅ Saved DocBin: {out_path} ({len(examples)} docs)")


def build_from_annotated(df, technical_seeds, soft_seeds):
    examples = []
    spans_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row['description'])
        ents = []

        # Case 1: Entities already annotated in JSON format
        if 'entities' in row and pd.notna(row['entities']):
            parsed = parse_entities_field(row['entities'])
            if parsed:
                doc = nlp(text)
                for ent in parsed:
                    s = int(ent['start'])
                    e = int(ent['end'])
                    ents.append((s, e, "SKILL"))
                    label = ent.get('meta_label') or guess_label(text[s:e], technical_seeds, soft_seeds)
                    spans_rows.append({'text': text[s:e], 'label': label})
                examples.append((doc, ents))
                continue

        # Case 2: skills column like "python:technical; communication:soft"
        if 'skills' in row and pd.notna(row['skills']):
            doc = nlp(text)
            tokens = str(row['skills']).split(";")
            for token in tokens:
                token = token.strip()
                if not token:
                    continue
                if ":" in token:
                    skill, lbl = token.split(":", 1)
                    skill = skill.strip()
                    lbl = lbl.strip()
                else:
                    skill = token
                    lbl = guess_label(skill, technical_seeds, soft_seeds)

                for s, e in find_span_offsets(text, skill):
                    ents.append((s, e, "SKILL"))
                    spans_rows.append({'text': text[s:e], 'label': lbl})
            if ents:
                examples.append((doc, ents))
                continue

        # Case 3: fallback seed-based weak supervision
        found = False
        doc = nlp(text)
        for phrase in list(technical_seeds) + list(soft_seeds):
            for s, e in find_span_offsets(text, phrase):
                ents.append((s, e, "SKILL"))
                lbl = 'technical' if phrase in technical_seeds else 'soft'
                spans_rows.append({'text': text[s:e], 'label': lbl})
                found = True

        if found:
            examples.append((doc, ents))

    return examples, pd.DataFrame(spans_rows)


def main(args):
    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    technical_seeds = read_seed_list(Path(args.technical_seed))
    soft_seeds = read_seed_list(Path(args.soft_seed))

    examples, spans_df = build_from_annotated(df, technical_seeds, soft_seeds)
    print(f"✅ Built {len(examples)} documents and {len(spans_df)} labeled skill spans")

    split = int(len(examples) * 0.9)
    train_exs = examples[:split]
    dev_exs = examples[split:]

    make_docbin(train_exs, args.out)
    make_docbin(dev_exs, args.dev)

    spans_df.to_csv(args.spans_csv, index=False)
    print(f"✅ Saved spans to: {args.spans_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--spans_csv", required=True)
    p.add_argument("--technical_seed", default="data/technical_skills.csv")
    p.add_argument("--soft_seed", default="data/soft_skills.csv")

    args = p.parse_args()
    main(args)
