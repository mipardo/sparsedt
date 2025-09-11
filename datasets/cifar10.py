from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from .utils import train_val_split, distribute_dataset


def get_loaders(dataset_config, batch_size, comm):
    """Get CIFAR-10 data loaders. 
    
    Args:
        dataset_config (utils.report.Dataset): dataset configuration parameters
        batch_size (int): batch size
        comm (MPI.Comm): MPI communicator (typically MPI.COMM_WORLD).
    
    TODO:
        If the data set is too large, it would be better to have process 0 load the entire data set and then distribute it to the other processes.
        
    Returns:
        tuple(train_loader, val_loader, test_loader): train_loader is divided among processes, val_loader and test_loader are the same for all procs
    """    
    train_transforms, test_transforms = _get_transforms()
    train_dataset, test_dataset = _get_dataset(train_transforms, test_transforms)
    train_dataset, val_dataset = train_val_split(train_dataset, dataset_config.val_split)
    train_dataset = distribute_dataset(train_dataset, comm)      
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def _get_transforms():
    train_transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    return train_transform, test_transform


def _get_dataset(train_transform, test_transform):
    train_dataset = datasets.CIFAR10(
        root="./datasets",
        train=True,
        download=True,
        transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root="./datasets",
        train=False,
        download=True,
        transform=test_transform
    )
    return train_dataset, test_dataset
