import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, num_tokens, num_classes, embed_dim=128, num_filters=100, kernel_sizes=(3, 4, 5)):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, embed_dim, padding_idx=0)
        self.min_len = max(kernel_sizes)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
    
    def forward(self, x):
        x = self.embedding(x).transpose(1,2)
        if x.size(2) < self.min_len:
            x = F.pad(x, (0, self.min_len - x.size(2)))
        pooled = [F.relu(conv(x)).max(dim=2).values for conv in self.convs]
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))

class TextLSTM(nn.Module):
    def __init__(self, num_tokens, num_classes, embed_dim=128, hidden_dim =128):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        lengths = (x != 0).sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            self.embedding(x), lengths, batch_first = True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(self.dropout(h))
        