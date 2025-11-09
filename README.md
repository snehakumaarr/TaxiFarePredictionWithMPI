# Stochastic Gradient Descent with MPI for predicting taxi fares
> Project for DSA5208 Scalable Distributed Computing for Data Science AY25/26 Sem 1

## Contributors
1. Sneha Kumar
2. Ryan Lee Ting Zhern
3. Mok Bingwei Maurice


### Directory structure
```
📦 project-root/
├── *📂 Data/*                              # (User-provided) Full dataset folder – user must download & unzip 'Data.zip' [here](https://drive.google.com/drive/folders/1zkSIGZ_tXeEEKRocKIIURtBo2TcA-SAt) or generate via data_preprocessing.ipynb
├── *📂 DataSplit/*                         # (User-provided) Pre-split datasets – user must download & unzip 'Data_Split.zip' [here](https://drive.google.com/drive/folders/1zkSIGZ_tXeEEKRocKIIURtBo2TcA-SAt) or generate via data_preprocessing.ipynb
├── 📂 InMemorySharding/                    # MPI Implementation with in-memory sharding
│   ├── *📂 results/*                       # (Auto-generated) created when sgd-train.py is run to store training/validation/test results and logs. 
│   ├── 🧮 activations.py                   # Python script containg activation functions
|   ├── 📜 dataset.py                       # Data script where root node reads the entire dataset to memory and distributes shards across other nodes
|   └── 🤖 sgd-train.py                     # Training script that runs SGD training with given configurations
├── 📂 MemoryEfficientSharding/             # MPI Implementation with memory efficient
│   ├── *📂 results/*                       # (Auto-generated) created when sgd-train.py is run to store training/validation/test results and logs. 
│   ├── 🧮 activations.py                   # Python script containg activation functions
|   ├── 📜 dataset.py                       # Data script where each node reads only its assigned shard (a subset of the data) from disk 
|   └── 🤖 sgd-train.py                     # Training script that runs SGD training with given configurations
├── 📂 PerformanceEvaluation/               # Notebooks for assessing and comparing model and MPI performance
│   ├── *📂 results/*                       # (Auto-generated) when bash script is run
│   ├── 📊 plot_comparison.ipynb            # Compares the training time and total task time for the two MPI implementations     
│   ├── ⏱️ process_time_plot.ipynb          # Compares the training time for different number of process 
│   ├── 🖥️ run_mpi_in_memory.sh             # Bash script to run In-Memory Sharing MPI implementation across different values of processes
│   ├── 💾 run_mpi_memory_efficient.sh      # Bash script to run Memory Efficient Sharing MPI implementation across different values of processes
|   └── 📉 training_curve.ipynb             # Plots training and validation curves from hyperparameter tuning
├── 📄 .gitignore
├── 📘 data_preprocessing.ipynb             # Notebook that contains data preprocessing steps (Warning: some notebook cells may take long)
├── 📄 README.md                            # You are here! (Project overview)
└── 📄 requirements.txt                     # Contains Python library requirements. 
```

## Environment 
You may set a virtual environment and install libraries based on the `requirements.txt`. 

## Data Preprocessing 
Clean data, separated into X and y train-valid-test can be directly accessed from [here](https://drive.google.com/drive/folders/1zkSIGZ_tXeEEKRocKIIURtBo2TcA-SAt). 
`Data.zip` contains 6 files (`X_train.npy`, `y_train.npy`, `X_valid.npy`, `y_valid.npy`, `X_test.npy`, `y_test.npy`). This is suitable for training with the In-Memory Sharding implementation. 
`Data_Split.zip` contains the data split into 2048 parts, for each X and y  train-valid-test. This is suitable for training with the Memory Efficient Sharding implementation. 

The scale for X features are standardized. y is not standardized/normalized (still on the original scale). 

The preprocess `data_preprocessing.ipynb` can also be run to generate the data from scratch for both In-Memory Sharding and Memory Efficient Sharding. The validation mode can be True or False to generate a dataset. When the validation mode is False, the dataset generated will be with train-test split of 70%-30% proportion. When the validation mode is True, the dataset generated will be with train-valid-test split of 60%-10%-30%. 

## Training
***Training Scripts*** <br/>
There are two folders: `InMemorySharding/`and `MemoryEfficientSharding/`. 
Each folder contains the same 3 scripts: 
`sgd-train.py` --> the main file used to train with MPI (configs can be passed via parameters, see below) <br/>
`activations.py` --> contains activation functions <br/>
`dataset.py` --> class for reading and splitting dataset across processes <br/>

Ensure you are in the desired directory, either `InMemorySharding/`or `MemoryEfficientSharding/`. 
### Basic Usage
```bash
# Example run with default parameters on 10 processes. 
mpiexec -n 10 python sgd-train.py
```

### Available Parameters
- `--data_directory`: Directory containing all the CSV files: X train, y train, X valid, y valid, X test, y test (default: "../Data/" for In Memory Sharding, and default: "../Data_Split" for Memory Efficient Sharding)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Mini-batch size per process (default: 256)
- `--learning_rate`: Learning rate for SGD (default: 0.003)
- `--n_hidden`: Number of hidden units (default: 32)
- `--activation`: Activation function - relu, sigmoid, leaky relu, tanh (default: "relu")
- `--split_dataset_size`: Only for Memory Efficient Sharding to specify how is the data pre-split (defual: 2048)
- `--log_every`: Log progress every N epochs (default: 5)
- `--seed`: To set seed for numpy and torch (default: 5208)
- `--validation_mode`: To specify if validation set is to be used or not (default: False). Setting this to True will mean that the script look for a validation set in your data directory. 


### With Custom Parameters
```bash
# Example with custom hyperparameters and 8 processes for single layer
mpiexec -n 8 python sgd-train.py \
  --data_directory "../Data/" \
  --epochs 100 \
  --batch_size 256 \
  --learning_rate 0.001 \
  --activation sigmoid \
  --log_every 1 \
  --n_hidden 256
```

```bash
# Example with custom hyperparameters and 8 processes for 3 layers with validation
mpiexec -n 8 python sgd-train.py \
  --data_directory "../Data_Split" \
  --epochs 100 \
  --batch_size 256 \
  --learning_rate 0.001 \
  --activation sigmoid \
  --log_every 10 \
  --validation_mode \
  --n_hidden 128 64 32 
```


### Hyperparameter Tuning Examples
```bash
# Test different learning rates
mpiexec -n 4 python sgd-train.py --learning_rate 0.1 --epochs 200
mpiexec -n 4 python sgd-train.py --learning_rate 0.01 --epochs 200
mpiexec -n 4 python sgd-train.py --learning_rate 0.001 --epochs 200

# Test different network sizes
mpiexec -n 4 python sgd-train.py --n_hidden 16 --epochs 100
mpiexec -n 4 python sgd-train.py --n_hidden 64 --epochs 100

# Test different activations
mpiexec -n 4 python sgd-train.py --activation relu --epochs 100
mpiexec -n 4 python sgd-train.py --activation sigmoid --epochs 100
```

**Note:** Results are automatically saved to `results/rmse_results_[shardingMethodName]_[numProcesses]_[hyperparameters].json` for later analysis and plotting. This results folder is created in the directory you are in. 

## Plotting and Analysis 
Naviagate to`PerformanceEvaluation/` to obtain the training curves as well compare training times for different processes and MPI implementation. 

You may use the `run_mpi_in_memory.sh` and `run_mpi_memory_efficient.sh` bash to run multiple commands at once using for loops. These bash scripts will create the `results\` folder in the `PerformanceEvaluation\` directory.  
