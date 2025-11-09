# sgd-train.py
"""
Main training script using PyTorch (MPS if available) + mpi4py.
- Uses Dataset which scatters shards from rank 0.
- Model is a one-hidden-layer network; gradients are computed on-device and averaged via MPI.
- get_batch now returns tensors already on DEVICE.
- No autograd changes requested; we keep manual gradient collection via loss.backward() and then average grads.
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpi4py import MPI
import numpy as np
import json
import os
import time

from dataset import Dataset
from activations import apply_activation

# Device selection
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

class HiddenNet(nn.Module):
    def __init__(self, m_features, n_hidden, activation_name):
        super().__init__()
        assert len(n_hidden) > 0, "There must be at least 1 Hidden Layer"
        self.hidden_layers = nn.ModuleList()
        for i in range(len(n_hidden)):
            if i == 0:
                self.hidden_layers.append(nn.Linear(m_features, n_hidden[i], bias = True))
            else:
                self.hidden_layers.append(nn.Linear(n_hidden[i-1], n_hidden[i], bias = True))
        self.output = nn.Linear(n_hidden[-1], 1, bias=True)
        self.activation_name = activation_name

    def forward(self, x):
        for i in range(len(self.hidden_layers)):
            x = self.hidden_layers[i](x)
            x = apply_activation(x, self.activation_name)
        x = self.output(x).squeeze(-1)
        return x 

# Flatten grads to numpy for MPI Allreduce
def flatten_grads_to_numpy(params):
    pieces = []
    for p in params:
        if p.grad is None:
            pieces.append(np.zeros(p.numel(), dtype=np.float32))
        else:
            g = p.grad.detach().cpu().float().numpy().reshape(-1)
            pieces.append(g)
    if len(pieces) == 0:
        return np.array([], dtype=np.float32)
    return np.concatenate(pieces).astype(np.float32)

# Unflatten numpy flat array back into param.grad (tensor on device)
def unflatten_numpy_to_grads(params, flat_numpy):
    idx = 0
    for p in params:
        n = p.numel()
        if n == 0:
            p.grad = None
            continue
        flat_slice = flat_numpy[idx:idx + n].reshape(p.shape)
        g_t = torch.from_numpy(flat_slice).to(p.device).view_as(p).clone()
        p.grad = g_t
        idx += n


def compute_rmse_mpi(model, X_local_numpy, y_local_numpy, mode, comm):
    assert mode in ["train", "val", "test"], "Mode Not Implemented"

    model.eval()
    with torch.no_grad():
        if X_local_numpy.shape[0] == 0:
            local_sum_sq = 0.0
            local_n = 0
        else:
            # Convert numpy arrays to tensors and move to device
            X_local = torch.from_numpy(X_local_numpy).to(DEVICE).float()
            y_local = torch.from_numpy(y_local_numpy).to(DEVICE).float().squeeze(-1)
            y_pred = model(X_local)
            local_sum_sq = float(((y_pred - y_local) ** 2).sum().item())
            local_n = int(X_local.shape[0])

    global_sum_sq = comm.allreduce(local_sum_sq, op=MPI.SUM)
    global_n = comm.allreduce(local_n, op=MPI.SUM)
    model.train()
    if global_n == 0:
        return float('nan')
    return float(np.sqrt(global_sum_sq / global_n))


def train(comm, args):
    # Start task timing 
    start_task_time = time.time()

    # Get rank and size
    rank = comm.Get_rank()
    size = comm.Get_size()
    print("rank, size:", rank, size)
    validation_mode = args.validation_mode

    # Load dataset
    print("Loading Datasets!")
    dataset = Dataset(args.data_directory, comm)
    m_features = dataset.X_train_local.shape[1]
    print("Datasets Loaded!")

     # Set seed for torch:
    torch.manual_seed(args.seed)

    model = HiddenNet(m_features, args.n_hidden, args.activation).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    mse_loss = nn.MSELoss(reduction='mean')

    # Lists to store RMSE results for each epoch
    train_rmse_history = []
    val_rmse_history = []
    epochs_logged = []

    # Start train timing
    start_train_time = time.time()
    if rank == 0:
        print(f"Starting training with {size} processes...")

    # Set seed for process
    np.random.seed(args.seed + rank)

    for epoch in range(args.epochs):
        X_batch, y_batch = dataset.get_batch(args.batch_size)
        # Convert numpy arrays to torch tensors and move to device
        X_batch = torch.from_numpy(X_batch).to(DEVICE).float()
        y_batch = torch.from_numpy(y_batch).to(DEVICE).float()

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = mse_loss(y_pred, y_batch.squeeze(-1))
        loss.backward()

        # Flatten grads to numpy (CPU), Allreduce (sum), average, and put back
        flat_local = flatten_grads_to_numpy(list(model.parameters()))
        flat_global = np.zeros_like(flat_local, dtype=np.float32)
        comm.Allreduce(flat_local, flat_global, op=MPI.SUM)
        flat_global /= size
        unflatten_numpy_to_grads(list(model.parameters()), flat_global)

        optimizer.step()

        if epoch % args.log_every == 0:
            train_rmse = compute_rmse_mpi(model, dataset.X_train_local, dataset.y_train_local, "train", comm)
            if validation_mode:
                val_rmse = compute_rmse_mpi(model, dataset.X_val_local, dataset.y_val_local, "val", comm)

            # Store RMSE results (only rank 0 needs to store, but all ranks compute for consistency)
            if rank == 0:
                train_rmse_history.append(train_rmse)
                epochs_logged.append(epoch)
                if validation_mode:
                    val_rmse_history.append(val_rmse)
                    print(f"[Epoch {epoch}] Train RMSE = {train_rmse:.6f} | Val RMSE = {val_rmse:.6f}")
                else:
                    print(f"[Epoch {epoch}] Train RMSE = {train_rmse:.6f}")

    # End train timing
    end_train_time = time.time()
    training_time = end_train_time - start_train_time

    # Final evaluation on test set (only after training is complete)
    final_test_rmse = compute_rmse_mpi(model, dataset.X_test_local, dataset.y_test_local, "test", comm)

    # End task timing
    end_task_time = time.time()
    total_task_time = end_task_time - start_task_time

    # Save RMSE results to file (only rank 0)
    if rank == 0:
        print(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        print(f"Final Test RMSE = {final_test_rmse:.6f}")
        
        if validation_mode:
            results = {
                'epochs': epochs_logged,
                'train_rmse': train_rmse_history,
                'val_rmse': val_rmse_history,  # During training validation RMSE
                'final_test_rmse': final_test_rmse,  # Final test RMSE after training
                'training_time_seconds': training_time,
                'training_time_minutes': training_time / 60,
                'total_task_time_seconds': total_task_time,
                'total_task_time_minutes': total_task_time / 60,
                'hyperparameters': {
                    'batch_size': args.batch_size,
                    'learning_rate': args.learning_rate,
                    'n_hidden': args.n_hidden,
                    'activation': args.activation,
                    'total_epochs': args.epochs,
                    'log_every': args.log_every,
                    'num_processes': size
                }
            }
        else:
            results = {
                'epochs': epochs_logged,
                'train_rmse': train_rmse_history,
                'final_test_rmse': final_test_rmse,  # Final test RMSE after training
                'training_time_seconds': training_time,
                'training_time_minutes': training_time / 60,
                'hyperparameters': {
                    'batch_size': args.batch_size,
                    'learning_rate': args.learning_rate,
                    'n_hidden': args.n_hidden,
                    'activation': args.activation,
                    'total_epochs': args.epochs,
                    'log_every': args.log_every,
                    'num_processes': size
                }
            }
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Generate filename with hyperparameters for easy identification
        fn_hidden = "_".join(list(map(lambda x: str(x), args.n_hidden)))
        filename = f"results/rmse_results_inmem_np{size}_lr{args.learning_rate}_bs{args.batch_size}_nh{fn_hidden}_{args.activation}_ep{args.epochs}"
        if validation_mode:
            filename = f"{filename}_with_validation.json"
        else:
            filename = f"{filename}.json"

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"RMSE results saved to: {filename}")
        print("Training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_directory", type=str, default="../Data/")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.003)
    parser.add_argument("--n_hidden", type=int, nargs= "+", default=[32])
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "sigmoid", "tanh", "leakyrelu"])
    parser.add_argument("--split_dataset_size", type=int, default=2048)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=5208)
    parser.add_argument("--validation_mode", action="store_true", default=False)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    train(comm, args)
