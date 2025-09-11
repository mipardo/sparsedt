import uuid
import subprocess


if __name__ == "__main__":
    test_id = f"benchmark-{str(uuid.uuid4())}"
    result_path = f"./running_logs/{test_id}.yaml"
    subprocess.run(["mpirun", "-np", "4", "python", "train_distributed.py", 
        "--config_file", "running_configs/test_mnist_oktopk.yaml",
        "--threads_per_worker", "2",
        "--verbose",
        "--tag", test_id
    ], check=True)