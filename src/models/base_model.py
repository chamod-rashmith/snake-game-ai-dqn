import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim: int = 32, action_dim: int = 3):
        """
        Deep Q-Network (DQN) for Reinforcement Learning.
        
        Args:
            state_dim (int): Dimension of the state observation space.
            action_dim (int): Number of discrete actions.
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Maps state/observation tensor to Q-values for each action.
        
        Args:
            state (torch.Tensor): State tensor of shape (batch_size, state_dim) or (state_dim,)
            
        Returns:
            torch.Tensor: Q-values of shape (batch_size, action_dim)
        """
        return self.network(state)

# Alias BaseModel to DQN for compatibility
BaseModel = DQN

if __name__ == "__main__":
    # Quick sanity check
    model = DQN(state_dim=32, action_dim=3)
    dummy_input = torch.randn(1, 32) # Batch size of 1, state_dim of 32
    q_values = model(dummy_input)
    print("DQN verification successful!")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output Q-values shape: {q_values.shape}")
    print(f"Output Q-values: {q_values.detach().numpy()}")
