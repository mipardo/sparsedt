import torch
import platform

def setup_device(running_env):
    """Setup and validate device"""
    if running_env == "cuda":
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Using CPU.")
            return torch.device("cpu")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        return torch.device("cuda")
    return torch.device("cpu")


def get_hardware_info(device, comm):
    hardware = {
        "process_id": 0,
        "torch_device": device,
        "host": platform.node(),
        "cpu_info": "Unknown",        
        "gpu_info": "Unknown",        
        "ram_info": "Unknown"        
    }

    if comm is not None:
        hardware["process_id"] = comm.rank
        return comm.allgather(hardware)
    
    return [hardware]
