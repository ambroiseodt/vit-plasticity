#!/usr/bin/bash

# Launch linear probing runs using apps/vit/linear_probing.py. It will create a dedicated
# tmux session on the specified device. To do so, run the following
# command in the terminal from the root directory of the project.
# ```bash
# $ bash <path_to_file_folder>/linear_probing.sh
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

# ---------------------------------
# Pretrained models on Imagenet-21K
# ---------------------------------

finetuned=False
cls_pooling=False
device="cuda:0"

# Common datasets
for dataset_name in \
    "cifar10" \
    "cifar100" \
    "pet" \
    "flowers102"
do
    # Runs
    session="lin_prob_${dataset_name}_pretrained"
    tmux new-session -d -s ${session}
    log_dir="vit_${dataset_name}_seed_0_lr_1e-3_comp_0"
    run="log_dir=${log_dir} finetuned=${finetuned} cls_pooling=${cls_pooling} dataset_name=${dataset_name} device=${device}"
    command="python -m apps.vit.linear_probing config=apps/vit/configs/linear_probing.yaml ${run}"
    echo "Running command: ${command}"
    tmux send-keys -t ${session} "${command}" C-m
done


# CIFAR10-C
for corruption in \
    "contrast" \
    "gaussian_noise" \
    "motion_blur" \
    "snow" \
    "speckle_noise"
do
    # Runs
    dataset_name="cifar10_c"
    session="lin_prob_${dataset_name}_${corruption}_pretrained"
    tmux new-session -d -s ${session}
    severity=5
    log_dir="vit_${dataset_name}_${corruption}_${severity}_seed_0_lr_1e-3_comp_0"
    run="log_dir=${log_dir} finetuned=${finetuned} cls_pooling=${cls_pooling} dataset_name=${dataset_name}-corruption-${corruption}-severity-${severity}  device=${device}"
    command="python -m apps.vit.linear_probing config=apps/vit/configs/linear_probing.yaml ${run}"
    echo "Running command: ${command}"
    tmux send-keys -t ${session} "${command}" C-m
done


# DOMAINNET
for domain in \
    "clipart" \
    "sketch"
do
    # Runs
    dataset_name="domainnet"
    session="lin_prob_${dataset_name}_${domain}_pretrained"
    tmux new-session -d -s ${session}
    log_dir="vit_${dataset_name}_${domain}_seed_0_lr_3e-3_comp_0"
    run="log_dir=${log_dir} finetuned=${finetuned} cls_pooling=${cls_pooling} dataset_name=${dataset_name}-${domain} device=${device}"
    command="python -m apps.vit.linear_probing config=apps/vit/configs/linear_probing.yaml ${run}"
    echo "Running command: ${command}"
    tmux send-keys -t ${session} "${command}" C-m
done