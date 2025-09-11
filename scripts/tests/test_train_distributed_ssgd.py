import uuid
import yaml
import subprocess


# This case test is not a real use case (distributed with no MPI), but it is design to assert basic funcionality
def test_distributed_training_on_mnist_1p_1th_no_MPI():
    test_id = f"test-{str(uuid.uuid4())}"
    result_path = f"./running_logs/{test_id}.yaml"
    subprocess.run(["python", "train_distributed.py",
        "--config_file", "running_configs/test_mnist_ssgd.yaml",
        "--tag", test_id
    ], check=True)

    with open(result_path, "r") as f:
        results = yaml.safe_load(f)

    assert results["running_conf"]["launch_params"]["environment"] == "cpu"
    assert results["running_conf"]["launch_params"]["model"]["name"] == "simplecnn"
    assert results["running_conf"]["launch_params"]["dataset"]["name"] == "mnist"
    assert results["running_conf"]["launch_params"]["optimizer"]["name"] == "ssgd"
    assert results["running_conf"]["launch_params"]["training"]["batch_size"] == 64
    assert results["distributed_conf"]["workers"] == 1
    assert results["distributed_conf"]["threads_per_worker"] == 1
    assert len(results["training"]["epochs"]) == 3
    assert results["testing"]["test_acc"] == 97.63
    assert results["training"]["train_epochs_cum_time"] < 23
    
    
# This case test is not a real use case (distributed with no MPI), but it is design to assert basic funcionality
def test_distributed_training_on_mnist_1p_2th_no_MPI():
    test_id = f"test-{str(uuid.uuid4())}"
    result_path = f"./running_logs/{test_id}.yaml"
    subprocess.run(["python", "train_distributed.py",
        "--threads_per_worker", "2",
        "--config_file", "running_configs/test_mnist_ssgd.yaml",
        "--tag", test_id
    ], check=True)

    with open(result_path, "r") as f:
        results = yaml.safe_load(f)

    assert results["running_conf"]["launch_params"]["environment"] == "cpu"
    assert results["running_conf"]["launch_params"]["model"]["name"] == "simplecnn"
    assert results["running_conf"]["launch_params"]["dataset"]["name"] == "mnist"
    assert results["running_conf"]["launch_params"]["optimizer"]["name"] == "ssgd"
    assert results["running_conf"]["launch_params"]["training"]["batch_size"] == 64
    assert results["distributed_conf"]["workers"] == 1
    assert results["distributed_conf"]["threads_per_worker"] == 2
    assert len(results["training"]["epochs"]) == 3
    assert results["testing"]["test_acc"] == 97.64
    assert results["training"]["train_epochs_cum_time"] < 23


# Single process and thread test 
def test_distributed_training_on_mnist_1p_1th():
    test_id = f"test-{str(uuid.uuid4())}"
    result_path = f"./running_logs/{test_id}.yaml"
    subprocess.run(["mpirun", "-np", "1", "python", "train_distributed.py",
        "--config_file", "running_configs/test_mnist_ssgd.yaml",
        "--tag", test_id
    ], check=True)

    with open(result_path, "r") as f:
        results = yaml.safe_load(f)

    assert results["running_conf"]["launch_params"]["environment"] == "cpu"
    assert results["running_conf"]["launch_params"]["model"]["name"] == "simplecnn"
    assert results["running_conf"]["launch_params"]["dataset"]["name"] == "mnist"
    assert results["running_conf"]["launch_params"]["optimizer"]["name"] == "ssgd"
    assert results["running_conf"]["launch_params"]["training"]["batch_size"] == 64
    assert results["distributed_conf"]["workers"] == 1
    assert results["distributed_conf"]["threads_per_worker"] == 1
    assert len(results["training"]["epochs"]) == 3
    assert results["testing"]["test_acc"] == 97.63
    assert results["training"]["train_epochs_cum_time"] < 23
    
    
# Single process and two-thread test 
def test_distributed_training_on_mnist_1p_2th():
    test_id = f"test-{str(uuid.uuid4())}"
    result_path = f"./running_logs/{test_id}.yaml"
    subprocess.run(["mpirun", "-np", "1", "python", "train_distributed.py",
        "--threads_per_worker", "2",
        "--config_file", "running_configs/test_mnist_ssgd.yaml",
        "--tag", test_id
    ], check=True)

    with open(result_path, "r") as f:
        results = yaml.safe_load(f)

    assert results["running_conf"]["launch_params"]["environment"] == "cpu"
    assert results["running_conf"]["launch_params"]["model"]["name"] == "simplecnn"
    assert results["running_conf"]["launch_params"]["dataset"]["name"] == "mnist"
    assert results["running_conf"]["launch_params"]["optimizer"]["name"] == "ssgd"
    assert results["running_conf"]["launch_params"]["training"]["batch_size"] == 64
    assert results["distributed_conf"]["workers"] == 1
    assert results["distributed_conf"]["threads_per_worker"] == 2
    assert len(results["training"]["epochs"]) == 3
    assert results["testing"]["test_acc"] == 97.64
    assert results["training"]["train_epochs_cum_time"] < 23
    

# Two-process and single thread test 
def test_distributed_training_on_mnist_2p_1th():
    test_id = f"test-{str(uuid.uuid4())}"
    result_path = f"./running_logs/{test_id}.yaml"
    subprocess.run(["mpirun", "-np", "2", "python", "train_distributed.py",
        "--config_file", "running_configs/test_mnist_ssgd.yaml",
        "--tag", test_id
    ], check=True)

    with open(result_path, "r") as f:
        results = yaml.safe_load(f)

    assert results["running_conf"]["launch_params"]["environment"] == "cpu"
    assert results["running_conf"]["launch_params"]["model"]["name"] == "simplecnn"
    assert results["running_conf"]["launch_params"]["dataset"]["name"] == "mnist"
    assert results["running_conf"]["launch_params"]["optimizer"]["name"] == "ssgd"
    assert results["running_conf"]["launch_params"]["training"]["batch_size"] == 64
    assert results["distributed_conf"]["workers"] == 2
    assert results["distributed_conf"]["threads_per_worker"] == 1
    assert len(results["training"]["epochs"]) == 3
    assert results["testing"]["test_acc"] == 95.89
    assert results["training"]["train_epochs_cum_time"] < 15


# Two-process and two-thread test 
def test_distributed_training_on_mnist_2p_2th():
    test_id = f"test-{str(uuid.uuid4())}"
    result_path = f"./running_logs/{test_id}.yaml"
    subprocess.run(["mpirun", "-np", "2", "python", "train_distributed.py",
        "--config_file", "running_configs/test_mnist_ssgd.yaml",
        "--threads_per_worker", "2",
        "--tag", test_id
    ], check=True)

    with open(result_path, "r") as f:
        results = yaml.safe_load(f)

    assert results["running_conf"]["launch_params"]["environment"] == "cpu"
    assert results["running_conf"]["launch_params"]["model"]["name"] == "simplecnn"
    assert results["running_conf"]["launch_params"]["dataset"]["name"] == "mnist"
    assert results["running_conf"]["launch_params"]["optimizer"]["name"] == "ssgd"
    assert results["running_conf"]["launch_params"]["training"]["batch_size"] == 64
    assert results["distributed_conf"]["workers"] == 2
    assert results["distributed_conf"]["threads_per_worker"] == 2
    assert len(results["training"]["epochs"]) == 3
    assert results["testing"]["test_acc"] == 95.89
    assert results["training"]["train_epochs_cum_time"] < 15
    
    
# Four-process and one-thread test 
def test_distributed_training_on_mnist_4p_1th():
    test_id = f"test-{str(uuid.uuid4())}"
    result_path = f"./running_logs/{test_id}.yaml"
    subprocess.run(["mpirun", "-np", "4", "python", "train_distributed.py",
        "--config_file", "running_configs/test_mnist_ssgd.yaml",
        "--tag", test_id
    ], check=True)

    with open(result_path, "r") as f:
        results = yaml.safe_load(f)

    assert results["running_conf"]["launch_params"]["environment"] == "cpu"
    assert results["running_conf"]["launch_params"]["model"]["name"] == "simplecnn"
    assert results["running_conf"]["launch_params"]["dataset"]["name"] == "mnist"
    assert results["running_conf"]["launch_params"]["optimizer"]["name"] == "ssgd"
    assert results["running_conf"]["launch_params"]["training"]["batch_size"] == 64
    assert results["distributed_conf"]["workers"] == 4
    assert results["distributed_conf"]["threads_per_worker"] == 1
    assert len(results["training"]["epochs"]) == 3
    assert results["testing"]["test_acc"] == 95.16
    assert results["training"]["train_epochs_cum_time"] < 11
    

# Four-process and two-thread test 
def test_distributed_training_on_mnist_4p_2th():
    test_id = f"test-{str(uuid.uuid4())}"
    result_path = f"./running_logs/{test_id}.yaml"
    subprocess.run(["mpirun", "-np", "4", "python", "train_distributed.py",
        "--config_file", "running_configs/test_mnist_ssgd.yaml",
        "--threads_per_worker", "2",
        "--tag", test_id
    ], check=True)

    with open(result_path, "r") as f:
        results = yaml.safe_load(f)

    assert results["running_conf"]["launch_params"]["environment"] == "cpu"
    assert results["running_conf"]["launch_params"]["model"]["name"] == "simplecnn"
    assert results["running_conf"]["launch_params"]["dataset"]["name"] == "mnist"
    assert results["running_conf"]["launch_params"]["optimizer"]["name"] == "ssgd"
    assert results["running_conf"]["launch_params"]["training"]["batch_size"] == 64
    assert results["distributed_conf"]["workers"] == 4
    assert results["distributed_conf"]["threads_per_worker"] == 2
    assert len(results["training"]["epochs"]) == 3
    assert results["testing"]["test_acc"] == 95.08
    assert results["training"]["train_epochs_cum_time"] < 11
        
