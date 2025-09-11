from torch.utils.data import random_split, Subset


def train_val_split(train_dataset, val_split):
    num_samples = len(train_dataset)
    val_samples = int(num_samples * val_split)
    train_samples = num_samples - val_samples 
    train, val = random_split(train_dataset, [train_samples, val_samples])
    return train, val


def distribute_dataset(dataset, comm):
    if comm is None:
        return dataset
    
    total_size = len(dataset)
    indices = list(range(total_size))
    subset_size = total_size // comm.size
    subset_start = subset_size * comm.rank
    subset_finish = subset_start + subset_size
    if comm.rank == comm.size - 1:
        subset_finish = total_size 
    return Subset(dataset, indices[subset_start: subset_finish])

