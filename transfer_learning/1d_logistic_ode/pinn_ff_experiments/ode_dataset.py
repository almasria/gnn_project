import torch 
from torch.utils.data import Dataset
import numpy as np


class ODEData(Dataset):
    def __init__(self, t_range, rho_values, t_points, constant_x, device='cuda:0'):
        """
        Initialize the dataset for a logistic growth ODE with a constant spatial coordinate.

        Args:
            t_range (list): Time domain [t_min, t_max].
            rho_values (list): List of reaction coefficients (? values).
            t_points (int): Number of time points.
            constant_x (float): The constant spatial coordinate (e.g., a representative location).
            device (str): Device to store the tensors ('cpu' or 'cuda:0').
        """
        self.device = device
        self.t_range = t_range
        self.rho_values = rho_values
        self.t_points = t_points
        self.constant_x = constant_x

        # Prepare data for each rho value.
        self.data = {}
        for rho in rho_values:
            # Generate residual points (time samples with constant x)
            res, ic = self._generate_data()
            res_tensor = torch.tensor(res, dtype=torch.float32, requires_grad=True).to(self.device)
            ic_tensor = torch.tensor(ic, dtype=torch.float32, requires_grad=True).to(self.device)
            
            # Precompute analytical solution at the initial condition (t = t_range[0])
            u_ic = self.analytical_solution(
                torch.tensor([[constant_x]], dtype=torch.float32, requires_grad=True).to(self.device),
                torch.tensor([[t_range[0]]], dtype=torch.float32, requires_grad=True).to(self.device),
                rho
            )
            
            self.data[rho] = {
                'res': res_tensor,   # (x, t) pairs over the time domain (x is constant)
                'ic': ic_tensor,     # Initial condition point (t = t_range[0])
                'u_ic': u_ic         # Analytical solution at t = t_range[0]
            }

    def _generate_data(self):
        """
        Generate residual points (for the interior of the time domain) and the initial condition.
        
        Returns:
            res (np.ndarray): Array of shape (t_points, 2) where each row is [constant_x, t].
            ic (np.ndarray): Array of shape (1, 2) corresponding to the initial condition at t = t_range[0].
        """
        # Create time samples
        t = np.linspace(self.t_range[0], self.t_range[1], self.t_points)
        # For each t, x is always the constant value provided.
        x = self.constant_x * np.ones_like(t)
        # Stack x and t to create our (x,t) pairs.
        res = np.stack([x, t], axis=-1)  # Shape: (t_points, 2)
        # The initial condition is simply the first row.
        ic = res[0:1, :]
        return res, ic

    def analytical_solution(self, x, t, rho):
        """
        Compute the analytical solution for the logistic growth ODE.
        Here we use the same functional form as before:
        
        u(t) = h(x) * exp(? t) / (h(x) * exp(? t) + 1 - h(x)),  with
        h(x) = exp( - (x - ?)? / [2*(?/4)?] ).
        
        Note: Since x is constant, h(x) is also constant.

        Args:
            x (torch.Tensor): The spatial input (constant value).
            t (torch.Tensor): Time input.
            rho (float): Reaction coefficient.
            
        Returns:
            torch.Tensor: The analytical solution.
        """
        pi = torch.tensor(np.pi, dtype=torch.float32, device=self.device)
        h = torch.exp(- (x - pi)**2 / (2 * (pi / 4)**2))
        return h * torch.exp(rho * t) / (h * torch.exp(rho * t) + 1 - h)

    def get_interior_points(self, rho):
        """
        Retrieve the interior (residual) points for a given rho.
        
        Returns:
            x (torch.Tensor): Spatial component (constant).
            t (torch.Tensor): Temporal component.
            rho_tensor (torch.Tensor): Tensor filled with the rho value.
        """
        res = self.data[rho]['res']
        x = res[:, 0:1]
        t = res[:, 1:2]
        rho_tensor = torch.full_like(x, rho)
        return x, t, rho_tensor

    def get_initial_condition(self, rho):
        """
        Retrieve the initial condition point and its analytical solution.
        
        Returns:
            x_ic, t_ic (torch.Tensor): The initial (x, t) point.
            rho_tensor (torch.Tensor): Tensor filled with the rho value.
            u_ic (torch.Tensor): The precomputed analytical solution at t = t_range[0].
        """
        ic = self.data[rho]['ic']
        x_ic = ic[:, 0:1]
        t_ic = ic[:, 1:2]
        rho_tensor = torch.full_like(x_ic, rho)
        u_ic = self.data[rho]['u_ic']
        return x_ic, t_ic, rho_tensor, u_ic

    def get_test_points(self, rho):
        """
        For this simple ODE experiment, the test points are the same as the interior points.
        
        Returns:
            x, t, rho tensor.
        """
        return self.get_interior_points(rho) 
    
    def get_interior_input_without_points(self):

        res, i_ = self._generate_data()
        res_tensor = torch.tensor(res, dtype=torch.float32, requires_grad=True).to(self.device)
            
        x = res_tensor[:, 0:1]
        t = res_tensor[:, 1:2]

        return x, t, None
        
