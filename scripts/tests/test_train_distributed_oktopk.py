import uuid
import yaml
import subprocess

    

# Four-process and two-thread test 
def test_distributed_training_on_mnist_4p_2th():
    test_id = f"test-{str(uuid.uuid4())}"
    result_path = f"./running_logs/{test_id}.yaml"
    subprocess.run(["mpirun", "-np", "4", "python", "train_distributed.py",
        "--config_file", "running_configs/test_mnist_oktopk.yaml",
        "--threads_per_worker", "2",
        "--tag", test_id
    ], check=True)

    with open(result_path, "r") as f:
        results = yaml.safe_load(f)

    assert results["running_conf"]["launch_params"]["environment"] == "cpu"
    assert results["running_conf"]["launch_params"]["model"]["name"] == "simplecnn"
    assert results["running_conf"]["launch_params"]["dataset"]["name"] == "mnist"
    assert results["running_conf"]["launch_params"]["optimizer"]["name"] == "oktopk"
    assert results["running_conf"]["launch_params"]["training"]["batch_size"] == 64
    assert results["distributed_conf"]["workers"] == 4
    assert results["distributed_conf"]["threads_per_worker"] == 2
    assert len(results["training"]["epochs"]) == 3
    assert results["testing"]["test_acc"] == 95.08
    assert results["training"]["train_epochs_cum_time"] < 11
        
