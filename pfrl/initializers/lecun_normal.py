import numpy as np
import torch


def init_lecun_normal(tensor, scale=1.0):
    """Initializes the tensor with LeCunNormal."""
    fan_in = torch.nn.init._calculate_correct_fan(tensor, "fan_in")
    std = scale * np.sqrt(1.0 / fan_in)
    with torch.no_grad():
        return tensor.normal_(0, std)
