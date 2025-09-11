from .vgg16 import VGG16
from .simplecnn import SimpleCNN

def get_model(model_config):
    """Get model based on configuration."""
    name = model_config.name.lower()
    
    if name == "vgg16":
        return VGG16(model_config.num_classes)
    if name == "simplecnn":
        return SimpleCNN(model_config.num_classes)
    else:
        raise ValueError(f"Unknown model: {name}") 