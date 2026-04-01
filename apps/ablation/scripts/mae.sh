#!/usr/bin/bash

# Launch plasticity analysis using apps/vit/analysis.py. It will create a dedicated
# tmux session on the specified device. To do so, run the following
# command in the terminal from the root directory of the project.
# ```bash
# $ bash <path_to_file_folder>/mae.sh
# ```

dataset_name="cifar10"

# Base
model_size="base"
batch_size=128
n_steps=100
device="cuda:4"
session="mae_base"
tmux new-session -d -s ${session}
command="python -m apps.ablation.mae run --model_size ${model_size} --dataset_name ${dataset_name} --batch_size ${batch_size} --n_steps ${n_steps} --device ${device}"
tmux send-keys -t ${session} "${command}" C-m

# Large
model_size="large"
batch_size=128
n_steps=100
device="cuda:4"
session="mae_large"
tmux new-session -d -s ${session}
command="python -m apps.ablation.mae run --model_size ${model_size} --dataset_name ${dataset_name} --batch_size ${batch_size} --n_steps ${n_steps} --device ${device}"
tmux send-keys -t ${session} "${command}" C-m

# Huge
model_size="huge"
batch_size=128
n_steps=100
device="cuda:4"
session="mae_huge"
tmux new-session -d -s ${session}
command="python -m apps.ablation.mae run --model_size ${model_size} --dataset_name ${dataset_name} --batch_size ${batch_size} --n_steps ${n_steps} --device ${device}"
tmux send-keys -t ${session} "${command}" C-m