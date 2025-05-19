import torch
import torch.nn as nn
import torch.nn.functional as F

class BitLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight, -1, 1)

    def forward(self, x):
        with torch.no_grad():
            ternary_w = self.weight.sign()
            ternary_w[torch.abs(self.weight) < 0.33] = 0
        out = F.linear(x, ternary_w, self.bias)
        return out

class BitGLU(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.proj_u = BitLinear(dim, hidden_dim)
        self.proj_g = BitLinear(dim, hidden_dim)
        self.proj_out = BitLinear(hidden_dim, dim)

    def forward(self, x):
        g = torch.sigmoid(self.proj_g(x))
        u = F.silu(self.proj_u(x))
        return self.proj_out(g * u)

class MLGRU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.f_gate = BitLinear(dim, dim)
        self.c_proj = BitLinear(dim, dim)
        self.g_gate = BitLinear(dim, dim)
        self.out_proj = BitLinear(dim, dim)

    def forward(self, x, h_prev):
        f = torch.sigmoid(self.f_gate(x))
        c = F.silu(self.c_proj(x))
        h = f * h_prev + (1 - f) * c

        g = torch.sigmoid(self.g_gate(x))
        o = self.out_proj(g * h)
        return o, h
    
class BitModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rnn = MLGRU(dim)
        self.glu = BitGLU(dim)
        self.state = None

    def forward(self, x):
        if self.state is None:
            self.state = torch.zeros_like(x)
        o, self.state = self.rnn(x, self.state)
        return self.glu(o)
