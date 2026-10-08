import re
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

LABELS = ["ChatGPT", "Claude", "Gemini"]
MAX_LEN = 400
PAD, UNK = 0, 1

def load(path):
    df = pd.read_csv(path)
    df["y"] = df["Source"].map({l: i for i, l in enumerate(LABELS)})
    df["group"] = df["Question"].str.lower().str.strip()
    return df

def split(df, seed=42):
    kfold = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    folds = [te for _, te in kfold.split(df, df.y, df.group)]
    test, val = folds[0], folds[1]
    train = np.concatenate(folds[2:])
    return train, val, test

def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())

def build_vocab(texts, min_freq=2):
    c = Counter(tok for text in texts for tok in tokenize(text))
    itos = ["<pad>", "<unk>"] + [w for w, n in c.most_common() if n>= min_freq]
    return {w: i for i, w in enumerate(itos)}

def encode(texts, vocab, max_len=MAX_LEN):
    out = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, text in enumerate(texts):
        ids = [vocab.get(w, UNK) for w in tokenize(text)][:max_len]
        out[i, :len(ids)] = ids
    return out