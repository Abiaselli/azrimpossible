import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Modules ---

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


# --- Full Astro Block ---

class AstroTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, vocab_size, decay=0.9):
        super().__init__()
        self.self_eval = PredictiveSelfEvaluator(hidden_dim, vocab_size)
        self.comparator = SelfEvalComparator(vocab_size)
        self.feedback_gate = AstroFeedbackGate(hidden_dim)
        self.decay = decay

    def forward(self, x, expected_probs, astro_trace):
        adjusted_state, sim_probs = self.self_eval(x)
        comp_score = self.comparator(sim_probs, expected_probs)  # [B, S, 1]
        new_trace = self.decay * astro_trace + (1 - self.decay) * comp_score
        x = adjusted_state + self.feedback_gate(new_trace)
        return x, new_trace, sim_probs, comp_score


# --- Demo Inference Loop ---

# Parameters
batch_size, seq_len, hidden_dim, vocab_size = 2, 4, 16, 10
x = torch.randn(batch_size, seq_len, hidden_dim)
expected_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
expected_probs = F.one_hot(expected_ids, num_classes=vocab_size).float()
astro_trace = torch.zeros(batch_size, seq_len, 1)

# Module
astro_block = AstroTransformerBlock(hidden_dim, vocab_size)

# Run
x_out, updated_trace, sim_probs, comp_scores = astro_block(x, expected_probs, astro_trace)

import pandas as pd
import numpy as np
import ace_tools_open as tools

# Prepare visualization
def tensor_to_df(tensor, label):
    return pd.DataFrame(tensor.squeeze().detach().numpy(), columns=[f"{label}_{i}" for i in range(tensor.shape[-1])])

df_output = pd.concat([
    tensor_to_df(sim_probs[0], "sim_prob"),
    tensor_to_df(comp_scores[0], "comp_score"),
    tensor_to_df(updated_trace[0], "astro_trace")
], axis=1)

tools.display_dataframe_to_user(name="AstroTrace Feedback Cycle (Sample 0)", dataframe=df_output)
