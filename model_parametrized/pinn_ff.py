### ADJUSTED for transfer learning experiments!!! additional input for equation parameter added ###
### Also different activation function inputs enabled

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

    
class PINNff(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer, init_act_func, subseq_activ_func):
        """
        FLS model with SinAct activation in the first layer and Tanh in subsequent layers.
        
        Args:
            in_dim (int): Input dimension (should now be 3 for x, t, and rho).
            hidden_dim (int): Number of neurons in hidden layers.
            out_dim (int): Output dimension (e.g., 1 for scalar output).
            num_layer (int): Total number of layers in the network.
        """
        super(PINNff, self).__init__()

        _activ_func_mapping = {"tanh": nn.Tanh(), "gelu": nn.GELU(), "sin": SinAct()}

        if init_act_func in _activ_func_mapping.keys():
            initial_activation = _activ_func_mapping[init_act_func]
        else: 
            raise ValueError(f"Invalid init_act_fn '{init_act_func}'. Must be one of {list(_activ_func_mapping.keys())}.")

        if subseq_activ_func in ["tanh", "gelu"]:
            subseq_activation = _activ_func_mapping[subseq_activ_func]
        else:
            raise ValueError(f"Invalid subseq_activ_func '{subseq_activ_func}'. Must be one of {["tanh", "gelu"]}")
            
       

        layers = []
        for i in range(num_layer - 1):
            if i == 0:
                layers.append(nn.Linear(in_features=in_dim, out_features=hidden_dim))
                layers.append(initial_activation)  # First layer uses SinAct
            else:
                layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
                layers.append(subseq_activation)  # Subsequent layers use Tanh

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