from .ssgd import SSGD
from .oktopk import OkTopk
import torch.optim as optim


def get_optimizer(optimizer, params):
    """Get optimizer based on configuration."""
    name = optimizer.name.lower()
    optimizer_params = optimizer.params
    
    if name == "sgd":
        return optim.SGD(
            params,
            lr = optimizer_params["learning_rate"],
            momentum = optimizer_params["momentum"],
            weight_decay = optimizer_params["weight_decay"]
        )
        
    if name == "ssgd":
        return SSGD(
            params,
            lr = optimizer_params["learning_rate"],
            momentum = optimizer_params["momentum"],
            weight_decay = optimizer_params["weight_decay"]
        )
        
    if name == "oktopk":
        return OkTopk(
            params,
            lr = optimizer_params["learning_rate"],
            momentum = optimizer_params["momentum"],
            weight_decay = optimizer_params["weight_decay"],
            density=optimizer_params["density"],
            tau=optimizer_params["tau"],
            tau_prime=optimizer_params["tau_prime"]
        )
    else:
        raise ValueError(f'Unknown optimizer: {name}') 