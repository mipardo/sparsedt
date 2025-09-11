import os
import uuid
import yaml
import time
import torch
from utils.report import *
from utils.hardware import get_hardware_info


class Logger:
    def __init__(self,  launch_params, computed_params, launch_tag, verbose, comm, log_dir="./running_logs"):
        self.comm = comm
        self.times = dict()
        self.verbose = verbose
        os.makedirs(log_dir, exist_ok=True)
        self.report = self._init_report(launch_params, computed_params, launch_tag)
        file_name = self.report.metadata.timestamp if len(launch_tag) == 0 else launch_tag
        self.log_file = f"{log_dir}/{file_name}.yaml"
        yaml.add_representer(str, str_presenter)


    def _init_report(self, launch_params, computed_params, launch_tag):
        metadata = Metadata(
            tag=launch_tag, 
            date = datetime.now(), 
            id = str(uuid.uuid4()), 
            timestamp=int(time.time() * 1000), 
            scheme_version = "1.0.0"
        )
        
        hardware = get_hardware_info(computed_params.device, self.comm)
        
        running_conf = RunningConf(
            launch_params=launch_params, 
            computed_params=computed_params
        )

        distributed_conf = DistributedConf(
            workers = 1 if self.comm is None else self.comm.size,
            threads_per_worker=torch.get_num_threads()
        )
        
        return Report(
            metadata=metadata,
            hardware=hardware,
            running_conf=running_conf,
            distributed_conf=distributed_conf
        )


    def on_training_start(self):
        if self.comm is not None and self.comm.rank != 0:
            return
        
        if self.verbose:
            print("\n# RUNNING PARAMETERS:\n")
            print(yaml.dump(self.report.running_conf.model_dump(), None, sort_keys=False))
            print("# STARTED TRAINING:\n")
            
        self.times["training_start"] = time.time()
        

    def on_epoch_train_start(self, epoch):
        if self.comm is not None and self.comm.rank != 0:
            return

        self.report.training.epochs.append(EpochResult(epoch=epoch))
        self.times[f"epoch_{epoch}_start"] = time.time()


    def on_epoch_train_end(self, epoch, train_loss, train_acc):
        if self.comm is not None and self.comm.rank != 0:
            return
        
        self.times[f"epoch_{epoch}_train_end"] = time.time()
        self.report.training.epochs[epoch].train_time = self.times[f"epoch_{epoch}_train_end"] - self.times[f"epoch_{epoch}_start"]
        self.report.training.epochs[epoch].train_loss = train_loss
        self.report.training.epochs[epoch].train_acc = train_acc


    def on_epoch_val_end(self, epoch, val_loss, val_acc):
        if self.comm is not None and self.comm.rank != 0:
            return
        
        self.times[f"epoch_{epoch}_val_end"] = time.time()
        self.report.training.epochs[epoch].train_val_time = self.times[f"epoch_{epoch}_val_end"] - self.times[f"epoch_{epoch}_start"]
        self.report.training.epochs[epoch].val_loss = val_loss
        self.report.training.epochs[epoch].val_acc = val_acc


    def on_training_end(self):
        if self.comm is not None and self.comm.rank != 0:
            return
        
        self.times["training_end"] = time.time()
        epoch_train_times = []
        epoch_train_val_times = []
        for epoch in range(len(self.report.training.epochs)):
            epoch_train_times.append(self.report.training.epochs[epoch].train_time)
            epoch_train_val_times.append(self.report.training.epochs[epoch].train_val_time )
        self.report.training.train_epochs_cum_time=sum(epoch_train_times) 
        self.report.training.train_epochs_avg=sum(epoch_train_times) / len(self.report.training.epochs)
        self.report.training.train_val_time=self.times["training_end"] - self.times["training_start"]


    def on_testing_start(self):
        if self.comm is not None and self.comm.rank != 0:
            return

        if self.verbose:
            print("\n# STARTED TESTING:\n")
            
        self.times["testing_start"] = time.time()
        

    def on_testing_end(self, test_loss, test_acc, show_report=False, ):
        if self.comm is not None and self.comm.rank != 0 or not show_report:
            return

        self.times["testing_end"] = time.time()
        self.report.testing.test_time=self.times["testing_end"] - self.times["testing_start"]
        self.report.testing.test_loss = test_loss
        self.report.testing.test_acc = test_acc
        with open(self.log_file, "a") as outfile:
            yaml.dump(self.report.model_dump(), outfile, sort_keys=False, default_flow_style=False)

        if self.verbose:
            print("\n# RESULTS:\n")
            print(yaml.dump(self.report.model_dump(), None, sort_keys=False, default_flow_style=False))



def str_presenter(dumper, data):
    if "\n" in data:  
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)