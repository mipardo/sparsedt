import yaml
import torch
import argparse
from tqdm import tqdm
from models import get_model
from datasets import get_dataset
from optimizers import get_optimizer
from utils.hardware import setup_device
from loss_functions import get_loss_function
from utils.logger import Logger
from utils.report import LaunchParams, ComputedParams


def train(model, train_loader, val_loader, criterion, optimizer, device, logger, config):
    """Train the model and log the metrics
    
    Args:
        model (torch.nn.Module): the model to be trained
        train_loader (torch.utils.data.dataloader.DataLoader): the inputs and targets to train the model
        val_loader (torch.utils.data.dataloader.DataLoader): the inputs and targets to validate the model
        criterion (torch.nn.modules.loss): loss function (e.g., nn.CrossEntropyLoss)
        optimizer (torch.optim): optimizer used to update the model params (e.g., torch.optim.SGD)
        device (torch.device): the device to run the training 
        logger (utils.logger.Logger): the logger to store and report the metrics
        config (utils.report.LaunchParams): the running parameters to train the model
    """
    
    logger.on_training_start()
    for epoch in range(config.training.epochs):
        logger.on_epoch_train_start(epoch)
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        logger.on_epoch_train_end(epoch, train_loss, train_acc)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, "Validation")
        logger.on_epoch_val_end(epoch, val_loss, val_acc)
    logger.on_training_end()


def test(model, test_loader, criterion, device, logger, post_training=True):
    """Test the model and log the metrics

    Args:
        model (torch.nn.Module): the model to be tested
        test_loader (torch.utils.data.dataloader.DataLoader): the inputs and targets to test the model
        criterion (torch.nn.modules.loss): loss function (e.g., nn.CrossEntropyLoss)
        device (torch.device): the device to run the training
        logger (utils.logger.Logger): the logger to store and report the metrics
        post_training (bool): if the test is performed before (False) or after (True) training. Only for logging purposes.
    """    
    
    logger.on_testing_start()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, "Testing")
    logger.on_testing_end(test_loss, test_acc, show_report=post_training)
            

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train a single epoch

    Args:
        model (torch.nn.Module): the model to be trained
        train_loader (torch.utils.data.dataloader.DataLoader): the inputs and targets to train the model
        criterion (torch.nn.modules.loss): loss function (e.g., nn.CrossEntropyLoss)
        optimizer (torch.optim): optimizer used to update the model params (e.g., torch.optim.SGD)
        device (torch.device): the device to run the training 

    Returns:
        tuple(float, float): average loss per batch and accuracy
    """    
    
    # Set model in train mode and prepare batch 
    pbar = tqdm(train_loader, desc="Training") 
    total_loss, correct, total = 0, 0, 0
    model.train()
    
    # For every batch in the loader
    for batch_id, (inputs, targets) in enumerate(pbar):
        # Load inputs and targets to device
        inputs, targets = inputs.to(device), targets.to(device)
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # Backward pass and weights update
        loss.backward()
        optimizer.step()  
        # Compute metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        # Show metrics in progress bar
        pbar.set_postfix({
            "loss": total_loss / (batch_id + 1),
            "acc": 100.0 * correct / total
        })
        
    # Return average loss per batch and accuracy
    return total_loss / len(train_loader), 100.0 * correct / total


def evaluate(model, loader, criterion, device, desc):
    """Evaluate the model

    Args:
        model (torch.nn.Module): the model to be evaluated
        loader (torch.utils.data.dataloader.DataLoader): the inputs and targets to evaluate the model
        criterion (torch.nn.modules.loss): loss function (e.g., nn.CrossEntropyLoss)
        device (torch.device): the device to run the evaluation
        desc (str): progress bar description, typically "Validation" or "Testing"

    Returns:
        tuple (float, float): average loss and accuracy    
    """    
    
    # Set model in eval mode and prepare data 
    total_loss, correct, total = 0, 0, 0
    model.eval()
    
    # No grads to avoid gradients computations and increase performance 
    with torch.no_grad():
        
        # For every batch in the loader:
        pbar = tqdm(loader, desc=desc)
        for batch_id, (inputs, targets) in enumerate(pbar):
            # Load inputs and targets to device
            inputs, targets = inputs.to(device), targets.to(device)
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # Compute metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            # Show metrics in progress bar
            pbar.set_postfix({
                "loss": total_loss / (batch_id + 1),
                "acc": 100.0 * correct / total
            })

    # Return average loss and accuracy
    return total_loss / len(loader), 100.0 * correct / total


def compute_additional_params(launch_params, device, model, loss_function, optimizer, train_loader, val_loader, test_loader):
    local_train_samples = len(train_loader) * launch_params.training.batch_size
    train_samples = local_train_samples
    val_samples = len(val_loader) * launch_params.training.batch_size
    test_samples = len(test_loader) * launch_params.training.batch_size
    total_samples = train_samples + val_samples + test_samples

    computed_params = ComputedParams(
        model=str(model),
        loss_function=str(loss_function),
        optimizer=str(optimizer),
        device=device.type,
        master_train_samples=local_train_samples,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        total_samples=total_samples,
    )
    return computed_params


if __name__ == "__main__":
    # Load launch_params    
    parser = argparse.ArgumentParser(prog="train")
    parser.add_argument("--threads", type=int, default=1, help="Threads to be used")
    parser.add_argument("--verbose", type=bool, default=False, help="True to show results on terminal")
    parser.add_argument("--tag", type=str, default="", help="add an specific tag to the metadata and file log")
    parser.add_argument("--config_file", type=str, default="running_configs/default_mnist.yaml", help="path to config file")
    args = parser.parse_args()
    with open(args.config_file, "r") as input_conf_file:
        running_config_dict = yaml.safe_load(input_conf_file)
        launch_params = LaunchParams(**running_config_dict)
    launch_tag = args.tag
    verbose = args.verbose

    # Initialize device, model, optimizer, loss function, the dataset, and compute addiontional params
    torch.manual_seed(9)
    torch.set_num_threads(args.threads)
    device = setup_device(launch_params.environment)
    model = get_model(launch_params.model).to(device)
    loss_function = get_loss_function(launch_params.training)
    optimizer = get_optimizer(launch_params.optimizer, model.parameters())
    train_loader, val_loader, test_loader = get_dataset(launch_params.dataset, launch_params.training.batch_size, comm=None)
    computed_params = compute_additional_params(launch_params, device, model, loss_function, optimizer, train_loader, val_loader, test_loader)

    # Initialize the logger and start training
    logger = Logger(launch_params, computed_params, launch_tag, verbose, None)
    test(model, test_loader, loss_function, device, logger, post_training=False)
    train(model, train_loader, val_loader, loss_function, optimizer, device, logger, launch_params)
    test(model, test_loader, loss_function, device, logger, post_training=True)
    
