# dataset.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math

class Dataset:
    def __init__(self, 
                 data_directory, 
                 comm, 
                 train_size, 
                 test_size, 
                 split_dataset_size, 
                 validation_mode = False, 
                 val_size = None): 
        """
        Dataset class for distributed training with train-validation-test split (60-10-30).
        
        Args:
            data_directory: Directory containing the data CSV files
            comm: MPI communicator
        """
        self.comm = comm
        self.data_directory = data_directory
        self.validation_mode = validation_mode
        rank = comm.Get_rank()
        size = comm.Get_size()
        assert rank < size, "Rank >= Size which shldn't happen!"

        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

        # Simple Computations to obtain start and end rank for the train set
        # Note that endpoint is exclusive
        per_mpi_train = self.train_size // size
        self.train_start = rank * per_mpi_train
        self.train_end = (rank + 1) * per_mpi_train if rank < (size - 1) else self.train_size

        per_mpi_test = self.test_size // size
        self.test_start = rank * per_mpi_test
        self.test_end = (rank+1) * per_mpi_test if rank < (size - 1) else self.test_size

        if self.validation_mode:
            per_mpi_val = self.val_size // size
            self.val_start = rank * per_mpi_val
            self.val_end = (rank + 1) * per_mpi_val if rank < (size - 1) else self.val_size

        # The amount of row in each block
        self.split_dataset_size = split_dataset_size

        # Features of X:
        self.m_features = np.load(f"{self.data_directory}/X_train/trainX_batch_0.npy").shape[1]

        if self.validation_mode:
            print(f"(Blocks) Process {rank} handling TRAIN indices {self.train_start} to {self.train_end-1}, "
                f"VAL indices {self.val_start} to {self.val_end-1}, "
                f"TEST indices {self.test_start} to {self.test_end-1}", flush=True)
        else:
            print(f"(Blocks) Process {rank} handling TRAIN indices {self.train_start} to {self.train_end-1}, "
                f"TEST indices {self.test_start} to {self.test_end-1}", flush=True)

    def _shard(self, X, y, size, rank):
        n = len(X)
        counts = [n // size + (1 if i < n % size else 0) for i in range(size)]
        starts = np.cumsum([0] + counts[:-1])
        ends = np.cumsum(counts)
        start = starts[rank]
        end = ends[rank]
        return X[start:end], y[start:end], int(start), int(end)

    def get_batch(self, batch_size):
        """Return a random batch from local training shard as torch tensors (on CPU).
           The training code will move tensors to device."""
        num_to_load = int(math.ceil(batch_size / self.split_dataset_size))
        assert num_to_load >= 1, "Number to Load cannot be less than 1!"
        id_choice = np.arange(self.train_start, self.train_end)
        idx_to_load = np.random.choice(id_choice, size = num_to_load, replace = False)

        # Load the Dataset (Might be > batch size)
        train_batch_X, train_batch_y = [], []
        for idx in idx_to_load:
            batch_sample_X = np.load(f"{self.data_directory}/X_train/trainX_batch_{idx}.npy")
            batch_sample_y = np.load(f"{self.data_directory}/y_train/trainy_batch_{idx}.npy")
            train_batch_X.append(batch_sample_X), train_batch_y.append(batch_sample_y)
        train_batch_X = np.concatenate(train_batch_X, axis = 0)
        train_batch_y = np.concatenate(train_batch_y, axis = 0)
        assert len(train_batch_X.shape) == 2, "This shld be a 2D Array!"

        n_local_batch = len(train_batch_X)
        idx_batch = np.random.choice(n_local_batch, size = batch_size, replace = False)
        return train_batch_X[idx_batch], train_batch_y[idx_batch]

        # n_local = len(self.X_train_local)
        # if n_local == 0:
        #     # return empty arrays
        #     return np.zeros((0, self.X_train_local.shape[1]), dtype=np.float32), np.zeros((0, self.y_train_local.shape[1]), dtype=np.float32)
        # bs = min(batch_size, n_local)
        # idx = np.random.choice(n_local, size=bs, replace=False)
        # return self.X_train_local[idx], self.y_train_local[idx]
