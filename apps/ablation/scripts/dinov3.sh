#!/usr/bin/bash

# Launch plasticity analysis using apps/vit/analysis.py. It will create a dedicated
# tmux session on the specified device. To do so, run the following
# command in the terminal from the root directory of the project.
# ```bash
# $ bash <path_to_file_folder>/dinov3.sh
# ```

dataset_name="cifar10"

# 7B
model_size="vit7b16"
batch_size=64
n_steps=200
device="cuda:0"
session="7B"
tmux new-session -d -s ${session}
command="python -m apps.ablation.dinov3 run --model_size ${model_size} --dataset_name ${dataset_name} --batch_size ${batch_size} --n_steps ${n_steps} --device ${device}"
tmux send-keys -t ${session} "${command}" C-m