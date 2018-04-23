import torch
import torch.nn as nn
import torch.nn.functional as F

from .dan import DAN

class SiameseDAN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim=100, num_classes=2):
        super(SiameseDAN, self).__init__()
        self.dan = DAN(vocab_size, embedding_dim, hidden_dim, -1)
        
        self.out = nn.Linear(2 * hidden_dim, num_classes)
        self.norm_out = nn.BatchNorm1d(num_classes)

    def load_pretrained(self, pretrained_embeddings, mode='static'):
        self.dan.load_pretrained(pretrained_embeddings, mode)
        
    def forward(self, d, q):
        d = self.dan(d)
        q = self.dan(q)
        
        h1, h2 = d * q, d + q
        
        x = torch.cat((h1, h2), 1)
        x = self.out(x)
        x = self.norm_out(x)

        return F.log_softmax(x, dim=1)