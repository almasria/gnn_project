import torch 
from torch.utils.data import Dataset
import numpy as np

class PDEData(Dataset):
    def __init__(self, x_range, y_range, x_num, y_num, beta_values ,device='cuda:0'):
        """
        Initialize the dataset for a PDE problem.

        Args:
            x_range (list): Spatial domain [x_min, x_max].
            y_range (list): Temporal domain [t_min, t_max].
            x_num (int): Number of spatial points.
            y_num (int): Number of temporal points.
            device (str): Device to store the tensors ('cpu' or 'cuda:0').
        """
        self.device = device
        self.x_range = x_range
        self.y_range = y_range
        self.x_num = x_num
        self.y_num = y_num
        self.beta_values = beta_values
        self.data = {}


    def _generate_data(x_range, y_range, x_num, y_num):
        x = np.linspace(x_range[0], x_range[1], x_num)
        t = np.linspace(y_range[0], y_range[1], y_num)

        x_mesh, t_mesh = np.meshgrid(x,t)
        data = np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1)
        
        b_left = data[0,:,:] 
        b_right = data[-1,:,:]
        b_upper = data[:,-1,:]
        b_lower = data[:,0,:]
        res = data.reshape(-1,2)

        return res, b_left, b_right, b_upper, b_lower    