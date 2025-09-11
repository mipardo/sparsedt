import uuid
import yaml
import subprocess


# This case test is not a real use case (not distributed with MPI), but it is design to assert basic funcionality
def test_training_on_mnist_1p_1th_with_MPI():
    test_id = f"test-{str(uuid.uuid4())}"
    result_path = f"./running_logs/{test_id}.yaml"
    subprocess.run(["mpirun", "-np", "1", "python", "train.py",
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
    
    
# This case test is not a real use case (not distributed with MPI), but it is design to assert basic funcionality
def test_training_on_mnist_1p_2th_with_MPI():
    test_id = f"test-{str(uuid.uuid4())}"
    result_path = f"./running_logs/{test_id}.yaml"
    subprocess.run(["mpirun", "-np", "1", "python", "train.py",
        "--config_file", "running_configs/test_mnist_ssgd.yaml",
        "--threads", "2",
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


# Sequential training with a single thread
def test_training_on_mnist_1p_1th():
    test_id = f"test-{str(uuid.uuid4())}"
    result_path = f"./running_logs/{test_id}.yaml"
    subprocess.run(["python", "train.py",
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
    

# Sequential training with two threads
def test_training_on_mnist_1p_2th():
    test_id = f"test-{str(uuid.uuid4())}"
    result_path = f"./running_logs/{test_id}.yaml"
    subprocess.run(["python", "train.py",
        "--threads", "2",
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


