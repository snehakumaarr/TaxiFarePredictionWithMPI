# activations.py
import torch
import torch.nn.functional as F

def apply_activation(x, name="relu"):
    if name == "relu":
        return F.relu(x)
    elif name == "sigmoid":
        return torch.sigmoid(x)
    elif name == "tanh":
        return torch.tanh(x)
    elif name == 'leakyrelu':
        return F.leaky_relu(x)
    else:
        raise ValueError(f"Unsupported activation function: {name}. Please add the activation function in activations.py and configure the argparse in sgd-train.py")