import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super(MultiheadAttention, self).__init__()

        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # Q, K, V 
        self.values = nn.Linear(dim, dim, bias=False)
        self.keys = nn.Linear(dim, dim, bias=False)
        self.queries = nn.Linear(dim, dim, bias=False)
        self.fc_out = nn.Linear(dim, dim)

    def forward(self, x):
        #print(x.shape)
        N, set_size, _ = x.shape
        
        # Q, K, V
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #dtype = torch.Double
        x = x.to(device)
        x = x.float()
        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        # split to multi-head
        values = values.view(N, set_size, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        keys = keys.view(N, set_size, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        queries = queries.view(N, set_size, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # self attention
        attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = F.softmax(attention / (self.head_dim ** 0.5), dim=3)
        out = torch.einsum("nhqk,nkhd->nqhd", [attention, values]).reshape(N, set_size, -1)

        return self.fc_out(out)


