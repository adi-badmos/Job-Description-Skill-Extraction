import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def main(args):
    spans = pd.read_csv(args.spans_csv)
    spans['label'] = spans['label'].fillna('unknown')
    # filter unknown or keep them? we'll drop unknown for training classifier
    spans = spans[spans['label'].isin(['technical','soft'])]
    if spans.empty:
        raise ValueError("No labeled spans for classifier training. Provide labeled skill spans (technical/soft).")

    X = spans['text'].astype(str)
    y = spans['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=5000)),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    print(classification_report(y_test, preds))
    joblib.dump(pipe, args.model_out)
    print(f"Saved classifier to {args.model_out}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--spans_csv", required=True)
    p.add_argument("--model_out", required=True)
    args = p.parse_args()
    main(args)
