import copy
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from model import TextCNN, TextLSTM
from data import load, split, build_vocab, encode, LABELS

df = load("LLM_Data_Final.csv")
train, val, test = split(df)
vocab = build_vocab(df.Answer.iloc[train])

def prep(idx):
    return torch.tensor(encode(df.Answer.iloc[idx], vocab)), torch.tensor(df.y.iloc[idx].values)

Xtr, ytr = prep(train) 
Xva, yva = prep(val)
Xte, yte = prep(test)

def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(X).argmax(1)

def run(Model, seed):
    torch.manual_seed(seed)
    model = Model(len(vocab), len(LABELS))
    opt = torch.optim.Adam(model.parameters(), lr = 1e-3)
    loss_fn = nn.CrossEntropyLoss()
    best_acc, best_state, bad = 0, None, 0
    for epoch in range(30):
        model.train()
        for b in torch.randperm(len(Xtr)).split(32):
            opt.zero_grad()
            loss_fn(model(Xtr[b]), ytr[b]).backward()
            opt.step()
        acc = (predict(model, Xva) == yva).float().mean().item()
        if acc > best_acc: 
            best_acc, best_state, bad = acc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad == 5: 
                break
    model.load_state_dict(best_state)
    return predict(model, Xte).numpy()

for name, Model in [("CNN", TextCNN), ("LSTM", TextLSTM)]:
    preds = [run(Model, seed) for seed in range(3)]
    print(f"\n=== {name} ===")
    print("test accuracy per seed:", [round(float((p == yte.numpy()).mean()), 3) for p in preds])
    y_all, p_all = np.tile(yte.numpy(), len(preds)), np.concatenate(preds)
    print(classification_report(y_all, p_all, target_names=LABELS, digits=3))
    print("confusion matrix(rows = true, cols = predicted):\n", confusion_matrix(y_all, p_all))
            
