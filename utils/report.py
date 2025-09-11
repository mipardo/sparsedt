from datetime import datetime 
from pydantic import BaseModel, Field
from typing import List, Optional, Literal


# Metadata
class Metadata(BaseModel):
    id: str = Field()
    date: datetime = Field()
    timestamp: int = Field() 
    tag: Optional[str] = Field()
    scheme_version: str = Field()


# Hardware 
class HardwareItem(BaseModel):
    host: str = Field()
    process_id: int = Field(ge=0)
    torch_device: str = Field()
    cpu_info: str = Field()
    gpu_info: str = Field()
    ram_info: str = Field()


# Running configuration: Launch parameters
class Training(BaseModel):
    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=2, multiple_of=2)
    loss_function: Literal["cce"] = Field()
    checkpoint_interval: int = Field(ge=0)


class Model(BaseModel):
    name: Literal["simplecnn", "vgg16"] = Field()
    num_classes: int = Field(ge=0)


class Dataset(BaseModel):
    name: Literal["mnist", "cifar10"] = Field()
    val_split: float = Field(ge=0.0, le=1.0)


class Optimizer(BaseModel):
    name: Literal["sgd", "ssgd", "oktopk"] = Field()
    params: dict = Field()


class LaunchParams(BaseModel):
    environment: Literal["cpu", "gpu"] = Field()
    training: Training = Field()
    model: Model = Field()
    dataset: Dataset = Field()
    optimizer: Optimizer = Field()


# Running configuration: Computed Parameters
class ComputedParams(BaseModel):
    model: Optional[str] = Field(default=None)
    loss_function: Optional[str] = Field(default=None)
    optimizer: Optional[str] = Field(default=None)
    device: Optional[str] = Field(default=None)
    master_train_samples: Optional[int] = Field(default=None, ge=0)
    train_samples: Optional[int] = Field(default=None, ge=0)
    val_samples: Optional[int] = Field(default=None, ge=0)
    test_samples: Optional[int] = Field(default=None, ge=0)
    total_samples: Optional[int] = Field(default=None, ge=0)


# Running configuration 
class RunningConf(BaseModel):
    launch_params: LaunchParams = Field()
    computed_params: ComputedParams = Field(default=ComputedParams())


# Distributed configuration
class DistributedConf(BaseModel):
    workers: int = Field(ge=1)
    threads_per_worker: int = Field(ge=1)


# Training results
class EpochResult(BaseModel):
    epoch: int = Field(ge=0)
    train_loss: float = Field(default=0.0)
    train_acc: float = Field(default=0.0, ge=0.0, le=100.0)
    val_loss: float = Field(default=0.0)
    val_acc: float = Field(default=0.0, ge=0.0, le=100.0)
    train_time: float = Field(default=0.0, ge=0.0) 
    train_val_time: float = Field(default=0.0, ge=0.0) 


class TrainingResult(BaseModel):
    train_val_time: float = Field(default=0, ge=0)
    train_epochs_cum_time: float = Field(default=0, ge=0)
    train_epochs_avg: float = Field(default=0, ge=0)
    epochs: List[EpochResult] = Field(default=[])
    

# Testing results
class TestingResult(BaseModel):
    test_loss: float = Field(default=0.0)
    test_acc: float = Field(default=0.0, ge=0.0, le=100.0)
    test_time: float = Field(default=0, ge=0)


# Report
class Report(BaseModel):
    metadata: Metadata = Field()
    hardware: List[HardwareItem] = Field()
    running_conf: RunningConf = Field()
    distributed_conf: DistributedConf = Field()
    training: TrainingResult = Field(default=TrainingResult())
    testing: TestingResult = Field(default=TestingResult())
