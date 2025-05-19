import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -- Core Components (copied to integrate into container class) --

class PredictiveSelfEvaluator(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.sim_head = nn.Linear(hidden_dim, vocab_size)
        self.eval_head = nn.Sequential(
            nn.Linear(vocab_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.fusion_gate = nn.Linear(hidden_dim + 1, hidden_dim)

    def forward(self, hidden_state):
        simulated_logits = self.sim_head(hidden_state)
        simulated_probs = F.softmax(simulated_logits, dim=-1)
        rewards = self.eval_head(simulated_probs).squeeze(-1).unsqueeze(-1)
        augmented_input = torch.cat([hidden_state, rewards], dim=-1)
        adjusted_state = self.fusion_gate(augmented_input)
        return adjusted_state, simulated_probs


class SelfEvalComparator(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(vocab_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, predicted_probs, expected_probs):
        combined = torch.cat([predicted_probs, expected_probs], dim=-1)
        score = self.discriminator(combined)
        return score


class AstroFeedbackGate(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(1, hidden_dim)

    def forward(self, astro_trace):
        return self.gate(astro_trace)


class AstroSelfAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.trace_bias = nn.Linear(1, n_heads)

    def forward(self, x, astro_trace):
        B, S, _ = x.shape
        Q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        trace_mod = self.trace_bias(astro_trace)  # [B, S, H]
        trace_mod = trace_mod.permute(0, 2, 1).unsqueeze(-1)  # [B, H, S, 1]

        print(f"scores shape: {scores.shape}")
        print(f"trace_mod: {trace_mod.shape}")
        scores = scores + trace_mod

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, -1)

        return self.out_proj(attn_output), attn_weights


# -- Full Model Wrapper --

class GlialMemoryModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=32, n_heads=4, seq_len=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.attn = AstroSelfAttention(hidden_dim, n_heads)
        self.self_eval = PredictiveSelfEvaluator(hidden_dim, vocab_size)
        self.comparator = SelfEvalComparator(vocab_size)
        self.feedback_gate = AstroFeedbackGate(hidden_dim)
        self.proj = nn.Linear(hidden_dim, vocab_size)
        self.seq_len = seq_len
        self.decay = 0.9

    def forward(self, x, expected_probs, astro_trace):
        x = self.embedding(x)  # [B, S, H]
        x, _ = self.attn(x, astro_trace)
        adjusted_state, sim_probs = self.self_eval(x)
        comp_score = self.comparator(sim_probs, expected_probs)
        new_trace = self.decay * astro_trace + (1 - self.decay) * comp_score
        x = adjusted_state + self.feedback_gate(new_trace)
        logits = self.proj(x)
        return logits, new_trace


# Generate synthetic delayed copy memory task
def generate_memory_task_data(batch_size, seq_len, vocab_size, delay=2):
    """
    Generates sequences like [A, B, <PAD>, <PAD>, A, B] where A and B are tokens.
    delay: number of padding tokens between input and repeat
    """
    assert seq_len >= 2 * delay + 2, "Sequence length too short for delayed copy."

    inputs = torch.randint(1, vocab_size - delay, (batch_size, 2))  # two initial tokens
    pad = torch.zeros(batch_size, delay, dtype=torch.long)
    trailing = inputs.clone()

    sequences = torch.cat([inputs, pad, trailing], dim=1)
    targets = F.one_hot(sequences, num_classes=vocab_size).float()  # for cross entropy or comparison

    return sequences, targets

model = GlialMemoryModel(vocab_size=12)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(100):
    inputs, targets = generate_memory_task_data(batch_size=16, seq_len=6, vocab_size=12)
    astro_trace = torch.zeros(inputs.shape[0], inputs.shape[1], 1)

    logits, astro_trace = model(inputs, targets, astro_trace)
    loss = loss_fn(logits[:, -2:].reshape(-1, 12), inputs[:, -2:].reshape(-1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
