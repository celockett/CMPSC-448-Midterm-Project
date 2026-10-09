import copy
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from model import TextCNN
from data import load, build_vocab, encode, LABELS

df = load("LLM_Data_Final.csv")
df["domain"] = df.group.map(pd.read_csv("LLM_Domains.csv").set_index("question")["domain"])
assert df.domain.notna().all(), "some questions have no domain label"

def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(X).argmax(1)

def run(Xtr, ytr, Xva, yva, Xte, vocab_size, seed):
    torch.manual_seed(seed)
    model = TextCNN(vocab_size, len(LABELS))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    best_acc, best_state, bad = 0, None, 0
    for epoch in range(30):
        model.train()
        for b in torch.randperm(len(Xtr)).split(32):
            opt.zero_grad()
            loss_fn(model(Xtr[b]), ytr[b]).backward()
            opt.step()
        acc = (predict(model, Xva) == yva).float().mean().item()
        if acc > best_acc: best_acc, best_state, bad = acc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad == 5: break
    model.load_state_dict(best_state)
    return predict(model, Xte).numpy()

rows = []
for held in sorted(df.domain.unique()):
    test, pool = np.where(df.domain == held)[0], np.where(df.domain != held)[0]
    sub = df.iloc[pool]
    folds = [v for _, v in StratifiedGroupKFold(10, shuffle=True, random_state=42).split(sub, sub.y, sub.group)]
    val, train = pool[folds[0]], pool[np.concatenate(folds[1:])]
    vocab = build_vocab(df.Answer.iloc[train])
    prep = lambda i: (torch.tensor(encode(df.Answer.iloc[i], vocab)), torch.tensor(df.y.iloc[i].values))
    (Xtr, ytr), (Xva, yva), (Xte, yte) = prep(train), prep(val), prep(test)
    preds = [run(Xtr, ytr, Xva, yva, Xte, len(vocab), seed) for seed in range(3)]
    y_all, p_all = np.tile(yte.numpy(), 3), np.concatenate(preds)
    f1s = f1_score(y_all, p_all, average=None, labels=range(len(LABELS)))
    rows.append({"held-out domain": held, "n_test": len(test), "accuracy": accuracy_score(y_all, p_all), "macro F1": f1_score(y_all, p_all, average="macro"), **dict(zip(LABELS, f1s))})
    print(rows[-1], flush=True)

print("\nCNN trained on the other domains (per-LLM columns are F1):")
print(pd.DataFrame(rows).round(3).to_string(index=False))