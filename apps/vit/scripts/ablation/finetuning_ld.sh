#!/usr/bin/bash

# Launch finetuning runs using apps/vit/train.py. It will create a dedicated
# tmux session on the specified device. To do so, run the following
# command in the terminal from the root directory of the project.
# ```bash
# $ bash <path_to_file_folder>/finetuning.sh
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


# # CIFAR10-C
# session="fin_cifar10ldc"
# dataset_name="cifar10ld_c"
# corruption="motion_blur"
# severity=5
# device="cuda:6"

# # Runs
# tmux new-session -d -s ${session}
# for seed in \
#     0 
# do
#     for lr in \
#         1e-2 
#     do
#         for i in "${!comps[@]}"
#         do
#             # Skip the emb finetuning
#             if [[ "$i" == 1 ]]; then
#                 continue
#             fi
#             log_dir="ablation/vit_${dataset_name}_${corruption}_${severity}_seed_${seed}_lr_${lr}_comp_${i}"
#             run="log_dir=${log_dir} dataset_name=${dataset_name}-corruption-${corruption}-severity-${severity} seed=${seed} lr=${lr} ${comps[$i]} device=${device}"
#             command="python -m apps.vit.train config=apps/vit/configs/${dataset_name}.yaml ${run}"
#             echo "Running command: ${command}"
#             tmux send-keys -t ${session} "${command}" C-m
#         done
#     done
# done

# # CIFAR100
# session="fin_cifar100ld"
# dataset_name="cifar100ld"
# device="cuda:2"

# # Runs
# tmux new-session -d -s ${session}
# for seed in \
#     0 
# do
#     for lr in \
#         1e-2 
#     do
#         for i in "${!comps[@]}"
#         do
#             # Skip the emb finetuning
#             if [[ "$i" == 1 ]]; then
#                 continue
#             fi
#             log_dir="ablation/vit_${dataset_name}_seed_${seed}_lr_${lr}_comp_${i}"
#             run="log_dir=${log_dir} seed=${seed} lr=${lr} ${comps[$i]} device=${device}"
#             command="python -m apps.vit.train config=apps/vit/configs/${dataset_name}.yaml ${run}"
#             echo "Running command: ${command}"
#             tmux send-keys -t ${session} "${command}" C-m
#         done
#     done
# done

# # DOMAINET
# session="fin_domainnetld_clipart"
# dataset_name="domainld"
# domain="clipart"
# device="cuda:3"

# # Runs
# tmux new-session -d -s ${session}
# for seed in \
#     0 
# do
#     for lr in \
#         1e-2 
#     do
#         for i in "${!comps[@]}"
#         do
#             # Skip the emb finetuning
#             if [[ "$i" == 1 ]]; then
#                 continue
#             fi
#             log_dir="ablation/vit_${dataset_name}_${domain}_seed_${seed}_lr_${lr}_comp_${i}"
#             run="log_dir=${log_dir} dataset_name=${dataset_name}-${domain} seed=${seed} lr=${lr} ${comps[$i]} device=${device}"
#             command="python -m apps.vit.train config=apps/vit/configs/${dataset_name}.yaml ${run}"
#             echo "Running command: ${command}"
#             tmux send-keys -t ${session} "${command}" C-m
#         done
#     done
# done

# session="fin_domainnetld_sketch"
# dataset_name="domainld"
# domain="sketch"
# device="cuda:4"

# # Runs
# tmux new-session -d -s ${session}
# for seed in \
#     0 
# do
#     for lr in \
#         1e-2 
#     do
#         for i in "${!comps[@]}"
#         do
#             # Skip the emb finetuning
#             if [[ "$i" == 1 ]]; then
#                 continue
#             fi
#             log_dir="ablation/vit_${dataset_name}_${domain}_seed_${seed}_lr_${lr}_comp_${i}"
#             run="log_dir=${log_dir} dataset_name=${dataset_name}-${domain} seed=${seed} lr=${lr} ${comps[$i]} device=${device}"
#             command="python -m apps.vit.train config=apps/vit/configs/${dataset_name}.yaml ${run}"
#             echo "Running command: ${command}"
#             tmux send-keys -t ${session} "${command}" C-m
#         done
#     done
# done

# PET
session="fin_ptld"
dataset_name="ptld"
device="cuda:5"

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
            run="log_dir=${log_dir} seed=${seed} lr=${lr} ${comps[$i]} device=${device}"
            command="python -m apps.vit.train config=apps/vit/configs/${dataset_name}.yaml ${run}"
            echo "Running command: ${command}"
            tmux send-keys -t ${session} "${command}" C-m
        done
    done
done
