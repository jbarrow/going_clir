import torch.nn as nn
import torch.nn.functional as F

class DAN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes=-1):
        super(DAN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.hidden = nn.Linear(embedding_dim, hidden_dim)
        self.norm_hidden = nn.BatchNorm1d(hidden_dim)
        
        if num_classes > 0:
            self.out = nn.Linear(2 * hidden_dim, num_classes)
            self.norm_out = nn.BatchNorm1d(num_classes)
        else:
            self.out = None
            
    def forward(self, x):
        x = self.embedding(x).mean(dim=1)
        x = self.norm_hidden(F.sigmoid(self.hidden(x)))
        
        if self.out is not None:
            x = self.norm_out(self.out(x))
        
        return x
    
    def load_pretrained(self, pretrained_embeddings, mode='static'):
        """
        Load pretraind embeddings given as a matrix, and potentially prevent
        them from updating.
        """
        self.embedding.weight.data.copy_(pretrained_embeddings)
        self.embedding.weight.requires_grad = (mode == 'nonstatic')