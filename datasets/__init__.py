from .mnist import get_loaders as get_mnist_loaders
from .cifar10 import get_loaders as get_cifar10_loaders

def get_dataset(dataset_config, batch_size, comm):
    """Get dataset based on configuration."""
    name = dataset_config.name.lower()
    
    if name == 'mnist':
        return get_mnist_loaders(dataset_config, batch_size, comm)
    if name == 'cifar10':
        return get_cifar10_loaders(dataset_config, batch_size, comm)
    else:
        raise ValueError(f'Unknown dataset: {name}') 