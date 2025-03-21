### ADJUSTED for transfer learning experiments!!! additional input for equation parameter added ###

# baseline implementation of First Layer Sine
# paper: Learning in Sinusoidal Spaces with Physics-Informed Neural Networks
# link: https://arxiv.org/abs/2109.09338

import torch
import torch.nn as nn


class SinAct(nn.Module):
    def __init__(self):
            super(SinAct, self).__init__() 

    def forward(self, x):
        return torch.sin(x)

    
class FLS_params(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer):
        """
        FLS model with SinAct activation in the first layer and Tanh in subsequent layers.
        
        Args:
            in_dim (int): Input dimension (should now be 3 for x, t, and rho).
            hidden_dim (int): Number of neurons in hidden layers.
            out_dim (int): Output dimension (e.g., 1 for scalar output).
            num_layer (int): Total number of layers in the network.
        """
        super(FLS_params, self).__init__()

        layers = []
        for i in range(num_layer - 1):
            if i == 0:
                layers.append(nn.Linear(in_features=in_dim, out_features=hidden_dim))
                layers.append(SinAct())  # First layer uses SinAct
            else:
                layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
                layers.append(nn.Tanh())  # Subsequent layers use Tanh

        layers.append(nn.Linear(in_features=hidden_dim, out_features=out_dim))  # Final layer

        self.linear = nn.Sequential(*layers)

    def forward(self, x, t, rho):
        """
        Forward pass of the FLS model.
        
        Args:
            x (torch.Tensor): Spatial input.
            t (torch.Tensor): Temporal input.
            rho (torch.Tensor): Reaction coefficient input.
        
        Returns:
            torch.Tensor: Model output.
        """
        # Concatenate x, t, and rho along the last dimension
        src = torch.cat((x, t, rho), dim=-1)
        return self.linear(src)