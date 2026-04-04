#!/usr/bin/bash

# Launch evaluation using apps/vit/eval.py. It will create a dedicated
# tmux session on the specified device. To do so, run the following
# command in the terminal from the root directory of the project.
# ```bash
# $ bash <path_to_file_folder>/eval_ld.sh
# ```

# Define components to freeze
declare -a comps=("components"=[]
                 "components=["attn_norm","mha","ffn_norm","ffn_fc1","ffn_fc2"]"
                 "components=["emb","mha","ffn_norm","ffn_fc1","ffn_fc2"]"
                 "components=["emb","attn_norm","ffn_norm","ffn_fc1","ffn_fc2"]"
                 "components=["emb","attn_norm","mha","ffn_fc1","ffn_fc2"]"
                 "components=["emb","attn_norm","mha","ffn_norm","ffn_fc2"]"
                 "components=["emb","attn_norm","mha","ffn_norm","ffn_fc1"]"
                )

# CIFAR100
session="eval_cifar100ld"
dataset_name="cifar100ld"
device="cuda:4"

# Runs
tmux new-session -d -s ${session}
for seed in \
    0 
do
    for lr in \
        1e-2
    do
        for i in "${!comps[@]}"
        do
            # Skip the emb finetuning
            if [[ "$i" == 1 ]]; then
                continue
            fi
            log_dir="ablation/vit_${dataset_name}_seed_${seed}_lr_${lr}_comp_${i}"
            run="log_dir=${log_dir} device=${device}"
            command="python -m apps.vit.eval config=apps/vit/configs/eval.yaml ${run}"
            echo "Running command: ${command}"
            tmux send-keys -t ${session} "${command}" C-m
        done        
    done
done


# CIFAR10-C
dataset_name="cifar10_c"
corruption="motion_blur"
severity=5
session="eval_cifar10ldc"
device="cuda:4"

# Runs
tmux new-session -d -s ${session}
for seed in \
    0 
do
    for lr in \
        1e-2
    do
        for i in "${!comps[@]}"
        do
            # Skip the emb finetuning
            if [[ "$i" == 1 ]]; then
                continue
            fi
            log_dir="ablation/vit_${dataset_name}_${corruption}_${severity}_seed_${seed}_lr_${lr}_comp_${i}"
            run="log_dir=${log_dir} device=${device}"
            command="python -m apps.vit.eval config=apps/vit/configs/eval.yaml ${run}"
            echo "Running command: ${command}"
            tmux send-keys -t ${session} "${command}" C-m
        done        
    done
done

# DOMAINNET
dataset_name="domainnet"
domain="clipart"
session="eval_domainnetld_clipart"
device="cuda:4"

# Runs
tmux new-session -d -s ${session}
for seed in \
    0 
do
    for lr in \
        1e-2
    do
        for i in "${!comps[@]}"
        do
            # Skip the emb finetuning
            if [[ "$i" == 1 ]]; then
                continue
            fi
            log_dir="ablation/vit_${dataset_name}_${domain}_seed_${seed}_lr_${lr}_comp_${i}"
            run="log_dir=${log_dir} device=${device}"
            command="python -m apps.vit.eval config=apps/vit/configs/eval.yaml ${run}"
            echo "Running command: ${command}"
            tmux send-keys -t ${session} "${command}" C-m
        done        
    done
done

dataset_name="domainnet"
domain="sketch"
session="eval_domainnetld_sketch"
device="cuda:4"

# Runs
tmux new-session -d -s ${session}
for seed in \
    0 
do
    for lr in \
        1e-2
    do
        for i in "${!comps[@]}"
        do
            # Skip the emb finetuning
            if [[ "$i" == 1 ]]; then
                continue
            fi
            log_dir="ablation/vit_${dataset_name}_${domain}_seed_${seed}_lr_${lr}_comp_${i}"
            run="log_dir=${log_dir} device=${device}"
            command="python -m apps.vit.eval config=apps/vit/configs/eval.yaml ${run}"
            echo "Running command: ${command}"
            tmux send-keys -t ${session} "${command}" C-m
        done        
    done
done