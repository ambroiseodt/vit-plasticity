#!/usr/bin/bash

# Launch plasticity analysis using apps/vit/analysis.py. It will create a dedicated
# tmux session on the specified device. To do so, run the following
# command in the terminal from the root directory of the project.
# ```bash
# $ bash <path_to_file_folder>/dinov3.sh
# ```

dataset_name="cifar10"

# Small
model_size="vits16"
batch_size=512
n_steps=25
device="cuda:4"
session="small"
tmux new-session -d -s ${session}
command="python -m apps.ablation.dinov3 run --model_size ${model_size} --dataset_name ${dataset_name} --batch_size ${batch_size} --n_steps ${n_steps} --device ${device}"
tmux send-keys -t ${session} "${command}" C-m

# Huge
model_size="vith16plus"
batch_size=128
n_steps=100
device="cuda:4"
session="huge"
tmux new-session -d -s ${session}
command="python -m apps.ablation.dinov3 run --model_size ${model_size} --dataset_name ${dataset_name} --batch_size ${batch_size} --n_steps ${n_steps} --device ${device}"
tmux send-keys -t ${session} "${command}" C-m

# 7B
model_size="vit7b16"
batch_size=64
n_steps=200
device="cuda:4"
session="7B"
tmux new-session -d -s ${session}
command="python -m apps.ablation.dinov3 run --model_size ${model_size} --dataset_name ${dataset_name} --batch_size ${batch_size} --n_steps ${n_steps} --device ${device}"
tmux send-keys -t ${session} "${command}" C-m