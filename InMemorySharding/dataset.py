# dataset.py -- only the root node reads all the data then scatters to all other nodes
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, data_directory, comm, validation_mode = False): 
        """
        Dataset class for distributed training with train-validation-test split (60-10-30).
        Root rank (0) loads all CSVs and scatters shards to other ranks.
        
        Args:
            data_directory: Directory containing the data CSV files (expects X_train.csv, y_train.csv, etc.)
            comm: MPI communicator
        """
        self.comm = comm
        self.validation_mode = validation_mode
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Helper to compute counts and starts uniformly
        def compute_counts_starts(n, size):
            counts = [n // size + (1 if i < n % size else 0) for i in range(size)]
            starts = np.cumsum([0] + counts[:-1])
            return counts, starts

        # Root loads full datasets; others set to None
        if rank == 0:
            X_train_full = np.load(data_directory + 'X_train.npy')
            y_train_full = np.load(data_directory + 'y_train.npy')
            X_test_full  = np.load(data_directory + 'X_test.npy')
            y_test_full  = np.load(data_directory + 'y_test.npy')

            n_train = len(X_train_full)
            n_test  = len(X_test_full)

            if self.validation_mode:
                X_val_full   = np.load(data_directory + 'X_valid.npy')
                y_val_full   = np.load(data_directory + 'y_valid.npy')
                n_val   = len(X_val_full)
           
        else:
            X_train_full = y_train_full = X_val_full = y_val_full = X_test_full = y_test_full = None
            n_train = n_val = n_test = 0

        # Broadcast sizes
        n_train = comm.bcast(n_train, root=0)
        if self.validation_mode:
            n_val = comm.bcast(n_val, root=0)
        n_test = comm.bcast(n_test, root=0)


        # Compute counts & starts for each split (all ranks can derive starts)
        train_counts, train_starts = compute_counts_starts(n_train, size)
        if self.validation_mode:
            val_counts, val_starts = compute_counts_starts(n_val, size)
        test_counts, test_starts = compute_counts_starts(n_test, size)

        # Root builds shard lists; others pass None and receive their shard
        if rank == 0:
            train_shards_X = [X_train_full[s:e] for s, e in zip(train_starts, train_starts + np.array(train_counts))]
            train_shards_y = [y_train_full[s:e] for s, e in zip(train_starts, train_starts + np.array(train_counts))]
            if self.validation_mode:
                val_shards_X = [X_val_full[s:e]   for s, e in zip(val_starts,   val_starts   + np.array(val_counts))]
                val_shards_y = [y_val_full[s:e]   for s, e in zip(val_starts,   val_starts   + np.array(val_counts))]
            test_shards_X  = [X_test_full[s:e]  for s, e in zip(test_starts,  test_starts  + np.array(test_counts))]
            test_shards_y  = [y_test_full[s:e]  for s, e in zip(test_starts,  test_starts  + np.array(test_counts))]

        else:
            train_shards_X = train_shards_y = None
            if self.validation_mode:
                val_shards_X = val_shards_y = None
            test_shards_X = test_shards_y = None

        # Scatter shards
        self.X_train_local = comm.scatter(train_shards_X, root=0)
        self.y_train_local = comm.scatter(train_shards_y, root=0)
        if self.validation_mode:
            self.X_val_local   = comm.scatter(val_shards_X, root=0)
            self.y_val_local   = comm.scatter(val_shards_y, root=0)
        self.X_test_local  = comm.scatter(test_shards_X, root=0)
        self.y_test_local  = comm.scatter(test_shards_y, root=0)


        # Derive local indices
        self.train_start = int(train_starts[rank]); self.train_end = int(self.train_start + train_counts[rank])
        if self.validation_mode:
            self.val_start   = int(val_starts[rank]);   self.val_end   = int(self.val_start   + val_counts[rank])
        self.test_start  = int(test_starts[rank]);  self.test_end  = int(self.test_start  + test_counts[rank])

        # Print shard info (only rank 0 prints sizes)
        if rank == 0:
            if self.validation_mode:
                print("Train, Val, Test sizes (total):", n_train, n_val, n_test)
                print(f"Process {rank} handling TRAIN indices {self.train_start} to {self.train_end-1}, "
                      f"VAL indices {self.val_start} to {self.val_end-1}, "
                      f"TEST indices {self.test_start} to {self.test_end-1}", flush=True)
            else:
                print("Train, Test sizes (total):", n_train, n_test)
                print(f"Process {rank} handling TRAIN indices {self.train_start} to {self.train_end-1}, "
                      f"TEST indices {self.test_start} to {self.test_end-1}", flush=True)

    def get_batch(self, batch_size):
        """Return a random batch from local training shard as numpy arrays."""
        n_local = len(self.X_train_local)
        if n_local == 0:
            return (np.zeros((0, self.m_features), dtype=np.float32),
                    np.zeros((0, self.y_train_local.shape[1] if self.y_train_local.ndim > 1 else 1), dtype=np.float32))
        bs = min(batch_size, n_local)
        idx = np.random.choice(n_local, size=bs, replace=False)
        return self.X_train_local[idx], self.y_train_local[idx]
