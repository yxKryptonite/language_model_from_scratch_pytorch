import torch
from torch import nn
import numpy as np
import datetime as dt
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

# 'vanilla'
class vanilla_LSTM(nn.Module):
    def __init__(self, words_num, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.Embedding = nn.Embedding(num_embeddings=words_num, embedding_dim=input_size)
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.Linear = nn.Linear(hidden_size, words_num)


    def forward(self, data):
        data = self.Embedding(data)
        h0 = torch.zeros(self.num_layers, data.shape[0], self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, data.shape[0], self.hidden_size, device=device)
        data, (_, _) = self.LSTM(data, (h0, c0))
        out = self.Linear(data)
        return out