import torch.nn as nn


def get_loss_function(training_config):
    """Get dataset based on configuration."""
    name = training_config.loss_function
    
    if name == 'cce':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f'Unknown loss function: {name}') 