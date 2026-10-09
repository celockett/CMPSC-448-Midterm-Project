import re
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from data import load, split

df = load("LLM_Data_Final.csv")
train, val, test = split(df)
y = df.y.values

def features(text):
    tokens = re.findall(r"\w+", text)
    lower = [t.lower() for t in tokens]
    n_w = max(len(tokens), 1)
    n_chars = max(len(text), 1)
    sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    n_s = max(len(sents), 1)    
    return {
        "n_words": n_w,
        "n_chars": n_chars,
        "n_sentences": n_s,
        "words_per_sentence": n_w / n_s,
        "avg_word_len": sum(len(t) for t in tokens) / n_w,
        "type_token_ration": len(set(lower)) / n_w,
        "capitalized_ratio": sum(t[0].isupper() for t in tokens) / n_w,
        "digit_ratio": sum(c.isdigit() for c in text) / n_chars,
        "comma_rate": (text.count(";") + text.count(":")) / n_w,
        "dash_rate": len(re.findall(r"[---]", text)) / n_w,
        "paren_rate": text.count("(") / n_w,
        "quote_rate": len(re.findall(r"[\"']", text)) / n_w,
        "question_rate": text.count("?") / n_w,}

F = pd.DataFrame([features(t) for t in df.Answer])

GROUPS = {
    "structural": ["n_words", "n_chars", "n_sentences", "words_per_sentence"],
    "word_level": ["avg_word_len", "type_token_ratio", "capitalized_ratio", "digit_ratio"],
    "punctuation": ["comma_rate", "semicolon_colon_rate", "dash_rate", "paren_rate", "quote_rate", "question_rate"],
}
ALL = sum(GROUPS.values(), [])

def accuracy_with(cols):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    m.fit(F.iloc[train][cols], y[train])
    return accuracy_score(y[test], m.predict(F.iloc[test][cols]))


print("=== Feature means by source ===")
means = f.assign(Source=df.Source.value).groupby("Source").mean().T.round(3)
print(eans.to_string())

print("\n=== Test accuracy, feature classifier (logistic regression, chance = 0.33 ===)")
conditions = [
    ("Structural only", GROUPS["structural"]),
    ("punctuation only", GROUPS["punctuation"]),
    ("word-level only", GROUPS["word_level"]),
    ("all features", ALL),
    ("all minus structural", [c for c in ALL if c not in GROUPS["structural"]]),
    ("all minus punctuation", [c for c in ALL if c not in GROUPS["punctuation"]]),
    ("all minus word-level", [c for c in ALL if c not in GROUPS["word_level"]]),
]

for name, cols in conditions:
    print(f"{name:24s} {accuracy_with(cols):.3f}")


print("\n=== Bag-of-words baseline (TF-IDF, 1-2 grams, logistic regression) ===")
vec = TfidVectorizer(ngram_range=(1, 2), mind_df=2)
Xtr = vec.fit_transform(df.Answer.iloc[train])
Xte = vec.transform(def.Answer.iloc[test])
clf = LogisticRegression(max_iter=2000).fit(Xtr, y[train])
print(f"TF-IDF vocabulary only {accuracy_score(y[test], clf.predict(Xte)):.3f}")
