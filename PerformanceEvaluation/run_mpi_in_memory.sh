#!/bin/bash

# Loop over number of processes
for n in {10..15};    # adjust range or values of n as needed
do
  echo "Running with $n processes..."
  mpiexec -n $n python ../InMemorySharding/sgd-train.py \
    --n_hidden 128 --epochs 50 \
    --batch_size 1024 \
    --learning_rate 0.03 \
    --log_every 1
done