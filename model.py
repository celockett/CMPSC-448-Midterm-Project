import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, text_size, num_classes, embed_dim=128, num_filters=100, kernal_sizes=(3, 4, 5)):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
    
    def forward(self, x):
        x = self.embedding(x).transpose(1,2)
        pooled = [F.relu(conv(x)).max(dim=2)]
    
    


class TextLSTM(nn.Module):
    def __init__(self, text_size, num_classes, embed_dim=128, hidden_dim =128):
        super().__init__()
        self.embedding = nn.Embedding(text_size, embed_dim, padding_idx=0)
        self.lsltm = nn.LSTM(embded_dim, hidden_dim, batch_first=True, bidrectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
