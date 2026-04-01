#!/usr/bin/bash

# Launch finetuning runs using apps/vit/train.py. It will create a dedicated
# tmux session on the specified device. To do so, run the following
# command in the terminal from the root directory of the project.
# ```bash
# $ bash <path_to_file_folder>/components.sh
# ```

# Define components to freeze
comps="components=["emb","attn_norm","ffn_norm","ffn_fc2"]"

# CIFAR100
session="comps_cifar100"
dataset_name="cifar100"
device="cuda:4"

# Runs
tmux new-session -d -s ${session}
for seed in \
    0 
do
    for lr in \
        1e-3 \
        3e-3 \
        1e-2 \
        3e-2
    do
        log_dir="ablation/vit_${dataset_name}_seed_${seed}_lr_${lr}_comp_mha_fc1"
        run="log_dir=${log_dir} seed=${seed} lr=${lr} ${comps} device=${device}"
        command="python -m apps.vit.train config=apps/vit/configs/${dataset_name}.yaml ${run}"
        echo "Running command: ${command}"
        tmux send-keys -t ${session} "${command}" C-m
    done
done


# DOMAINNET
session="comps_domainnet_clipart"
dataset_name="domainnet"
domain="clipart"
device="cuda:2"

# Runs
tmux new-session -d -s ${session}
for seed in \
    0 
do
    for lr in \
        3e-3 \
        1e-2 \
        3e-2 \
        6e-2
    do
        log_dir="ablation/vit_${dataset_name}_${domain}_seed_${seed}_lr_${lr}_comp_mha_fc1"
        run="log_dir=${log_dir} dataset_name=${dataset_name}-${domain} seed=${seed} lr=${lr} ${comps} device=${device}"
        command="python -m apps.vit.train config=apps/vit/configs/${dataset_name}.yaml ${run}"
        echo "Running command: ${command}"
        tmux send-keys -t ${session} "${command}" C-m
    done
done
