#!/usr/bin/bash

# Launch plasticity analysis using apps/vit/analysis.py. It will create a dedicated
# tmux session on the specified device. To do so, run the following
# command in the terminal from the root directory of the project.
# ```bash
# $ bash <path_to_file_folder>/gpt2.sh
# ```

dataset_name="ag_news"

# Base
model_size="base"
seq_len=128
batch_size=512
n_steps=25
device="cuda:2"
session="gpt2"
tmux new-session -d -s ${session}
command="python -m apps.ablation.gpt2 run --model_size ${model_size} --dataset_name ${dataset_name} --batch_size ${batch_size} --seq_len ${seq_len} --n_steps ${n_steps} --device ${device}"
tmux send-keys -t ${session} "${command}" C-m
