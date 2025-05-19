import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the PredictiveSelfEvaluator class
class PredictiveSelfEvaluator(nn.Module):
    def __init__(self, hidden_dim, vocab_size, simulation_depth=1):
        super().__init__()
        self.sim_head = nn.Linear(hidden_dim, vocab_size)     # simulate output
        self.eval_head = nn.Sequential(                       # evaluate it
            nn.Linear(vocab_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # reward estimate between 0 and 1
        )
        self.fusion_gate = nn.Linear(hidden_dim + 1, hidden_dim)

    def forward(self, hidden_state):
        """
        hidden_state: [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = hidden_state.shape

        # Step 1: Predict token (simulated logit output)
        simulated_logits = self.sim_head(hidden_state)                    # [B, S, V]
        simulated_token_probs = F.softmax(simulated_logits, dim=-1)

        # Step 2: Evaluate simulation
        rewards = self.eval_head(simulated_token_probs).squeeze(-1)      # [B, S]
        rewards = rewards.unsqueeze(-1)                                  # [B, S, 1]

        # Step 3: Feedback into main pipeline
        augmented_input = torch.cat([hidden_state, rewards], dim=-1)     # [B, S, H+1]
        adjusted_state = self.fusion_gate(augmented_input)               # [B, S, H]

        return adjusted_state, simulated_logits.detach()

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
        """
        Compare simulated and actual token distributions
        """
        combined = torch.cat([predicted_probs, expected_probs], dim=-1)
        score = self.discriminator(combined)
        return score


# Dummy parameters
batch_size, seq_len, hidden_dim, vocab_size = 2, 4, 16, 10
dummy_hidden_state = torch.randn(batch_size, seq_len, hidden_dim)

# Simulate expected probs (e.g., real outputs or teacher forcing)
expected_token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
expected_probs = F.one_hot(expected_token_ids, num_classes=vocab_size).float()

# Modules
self_eval_layer = PredictiveSelfEvaluator(hidden_dim, vocab_size)
comparator = SelfEvalComparator(vocab_size)

# Run simulation + comparison loop
adjusted_state, simulated_probs = self_eval_layer(dummy_hidden_state)
comparison_scores = comparator(simulated_probs, expected_probs)

print(f"adjusted_state shape: {adjusted_state.shape}")
print(f"simulated_probss shape: {simulated_probs.shape}")
print(f"comparison_scores shape: {comparison_scores.shape} ")