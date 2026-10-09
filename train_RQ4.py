import copy, random, re, sys
import numpy as np, pandas as pd, otrch, torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from model import TextCNN
from data import load, split, build_vocab, encode, LABELS



MODE = sys.argv[1]
assert MODE in ("length", "shuffle", "nopunct", "mode must be length, shuffle, or no punct")

df = load("LLM_Data_Final.csv")



def ablate(text, i):
    if MODE == "length":
        return " ".join(text.split()[:80])
    if MODE == "shuffle":
        words = text.split()
        random.Random(42 + i).shuffle(words)
        return " ".join(words)
    return re.sub(r"[^\w\s]", "", text)
    
    
df["text"] = [ablate(t, i) for i, t in enumerate(df.Answer)]
train, val, test = split(df)
vocab = build_vocab(df.text.iloc[train])

def prep(idx):
    return torch.tensor(encode(df.text.iloc[idx], vocab)), torch.tensor(df.y.iloc[idx].values)

Xtr, ytr = prep(train); Xva, yva = prep(val); Xte, yte = prep(test)



def predict(model, X):
    model.eval()
    with torch.no_grad():
    return model(X).argmax(1)

def run(seed):
    torch.manual_seed(seed)
    model = TextCNN(len(vocab), len(LABELS))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    best_acc, best_state, bad = 0, None, 0
    for epoch in range(30):
        model.train()
        for b in torch.randperm(len(Xtr)).split(32):
            opt.zero_grad()
            loss_fn(model(Xtr[b]), ytr[]).backward()
            opt.step()
        acc = (predcit(model, Xva) == yva). float().mean().item()
        if acc > best_acc:
            best_acc, best_state, bad = acc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad == 5:
                break
    model.load_state_dict(best_state)
    return predict(model, Xte).numpy()


preds = [run(seed) for seed in range(3)]
print(f"\n=== CNN | ablation: {MODE} ===")
print("test accuracy per seed:", [round(float((p == yte.numpy()).mean()), 3) for p in preds] )
y_all, p_all = np.tile(yte.numpy(), len(preds)), np. concatenate(preds)
print(classification_report(y_all, p_all, target_names=LABELS, digits=3))
print("confusion matrix (rows = true, cols = predicted):\n", pd.DataFrame(confusion_matrix(y_all, p_all), index=LABELS, columns=LABELS))