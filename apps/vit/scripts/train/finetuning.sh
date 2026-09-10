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

# CIFAR10
session="fin_cifar10"
dataset_name="cifar10"
device="cuda:0"

# Runs
tmux new-session -d -s ${session}
for seed in \
    0 \
    42 \
    3407
do
    for lr in \
        1e-3 \
        3e-3 \
        1e-2 \
        3e-2
    do
        for i in "${!comps[@]}"
        do
            log_dir="vit_${dataset_name}_seed_${seed}_lr_${lr}_comp_${i}"
            run="log_dir=${log_dir} seed=${seed} lr=${lr} ${comps[$i]} device=${device}"
            command="python -m apps.vit.train config=apps/vit/configs/${dataset_name}.yaml ${run}"
            echo "Running command: ${command}"
            tmux send-keys -t ${session} "${command}" C-m
        done
    done
done

# CIFAR100
session="fin_cifar100"
dataset_name="cifar100"
device="cuda:0"

# Runs
tmux new-session -d -s ${session}
for seed in \
    0 \
    42 \
    3407
do
    for lr in \
        1e-3 \
        3e-3 \
        1e-2 \
        3e-2
    do
        for i in "${!comps[@]}"
        do
            log_dir="vit_${dataset_name}_seed_${seed}_lr_${lr}_comp_${i}"
            run="log_dir=${log_dir} seed=${seed} lr=${lr} ${comps[$i]} device=${device}"
            command="python -m apps.vit.train config=apps/vit/configs/${dataset_name}.yaml ${run}"
            echo "Running command: ${command}"
            tmux send-keys -t ${session} "${command}" C-m
        done
    done
done

# PET
session="fin_pet"
dataset_name="pet"
device="cuda:0"

# Runs
tmux new-session -d -s ${session}
for seed in \
    0 \
    42 \
    3407
do
    for lr in \
        1e-3 \
        3e-3 \
        1e-2 \
        3e-2
    do
        for i in "${!comps[@]}"
        do
            log_dir="vit_${dataset_name}_seed_${seed}_lr_${lr}_comp_${i}"
            run="log_dir=${log_dir} seed=${seed} lr=${lr} ${comps[$i]} device=${device}"
            command="python -m apps.vit.train config=apps/vit/configs/${dataset_name}.yaml ${run}"
            echo "Running command: ${command}"
            tmux send-keys -t ${session} "${command}" C-m
        done
    done
done

# FLOWERS102
session="fin_flowers102"
dataset_name="flowers102"
device="cuda:0"

# Runs
tmux new-session -d -s ${session}
for seed in \
    0 \
    42 \
    3407
do
    for lr in \
        1e-3 \
        3e-3 \
        1e-2 \
        3e-2
    do
        for i in "${!comps[@]}"
        do
            log_dir="vit_${dataset_name}_seed_${seed}_lr_${lr}_comp_${i}"
            run="log_dir=${log_dir} seed=${seed} lr=${lr} ${comps[$i]} device=${device}"
            command="python -m apps.vit.train config=apps/vit/configs/${dataset_name}.yaml ${run}"
            echo "Running command: ${command}"
            tmux send-keys -t ${session} "${command}" C-m
        done
    done
done


# CIFAR10-C
session="fin_cifar10_c_snow"
dataset_name="cifar10_c"
corruption="snow"
severity=5
device="cuda:0"

# Runs
tmux new-session -d -s ${session}
for seed in \
    0 \
    42 \
    3407
do
    for lr in \
        1e-3 \
        3e-3 \
        1e-2 \
        3e-2
    do
        for i in "${!comps[@]}"
        do
            log_dir="vit_${dataset_name}_${corruption}_${severity}_seed_${seed}_lr_${lr}_comp_${i}"
            run="log_dir=${log_dir} dataset_name=${dataset_name}-corruption-${corruption}-severity-${severity} seed=${seed} lr=${lr} ${comps[$i]} device=${device}"
            command="python -m apps.vit.train config=apps/vit/configs/${dataset_name}.yaml ${run}"
            echo "Running command: ${command}"
            tmux send-keys -t ${session} "${command}" C-m
        done
    done
done

# CIFAR10-C
session="fin_cifar10_c_contrast"
dataset_name="cifar10_c"
corruption="contrast"
severity=5
device="cuda:0"

# Runs
tmux new-session -d -s ${session}
for seed in \
    0 \
    42 \
    3407
do
    for lr in \
        1e-3 \
        3e-3 \
        1e-2 \
        3e-2
    do
        for i in "${!comps[@]}"
        do
            log_dir="vit_${dataset_name}_${corruption}_${severity}_seed_${seed}_lr_${lr}_comp_${i}"
            run="log_dir=${log_dir} dataset_name=${dataset_name}-corruption-${corruption}-severity-${severity} seed=${seed} lr=${lr} ${comps[$i]} device=${device}"
            command="python -m apps.vit.train config=apps/vit/configs/${dataset_name}.yaml ${run}"
            echo "Running command: ${command}"
            tmux send-keys -t ${session} "${command}" C-m
        done
    done
done

# CIFAR10-C
session="fin_cifar10_c_gaussian_noise"
dataset_name="cifar10_c"
corruption="gaussian_noise"
severity=5
device="cuda:0"

# Runs
tmux new-session -d -s ${session}
for seed in \
    0 \
    42 \
    3407
do
    for lr in \
        1e-3 \
        3e-3 \
        1e-2 \
        3e-2
    do
        for i in "${!comps[@]}"
        do
            log_dir="vit_${dataset_name}_${corruption}_${severity}_seed_${seed}_lr_${lr}_comp_${i}"
            run="log_dir=${log_dir} dataset_name=${dataset_name}-corruption-${corruption}-severity-${severity} seed=${seed} lr=${lr} ${comps[$i]} device=${device}"
            command="python -m apps.vit.train config=apps/vit/configs/${dataset_name}.yaml ${run}"
            echo "Running command: ${command}"
            tmux send-keys -t ${session} "${command}" C-m
        done
    done
done


# CIFAR10-C
session="fin_cifar10_c_motion_blur"
dataset_name="cifar10_c"
corruption="motion_blur"
severity=5
device="cuda:0"

# Runs
tmux new-session -d -s ${session}
for seed in \
    0 \
    42 \
    3407
do
    for lr in \
        1e-3 \
        3e-3 \
        1e-2 \
        3e-2
    do
        for i in "${!comps[@]}"
        do
            log_dir="vit_${dataset_name}_${corruption}_${severity}_seed_${seed}_lr_${lr}_comp_${i}"
            run="log_dir=${log_dir} dataset_name=${dataset_name}-corruption-${corruption}-severity-${severity} seed=${seed} lr=${lr} ${comps[$i]} device=${device}"
            command="python -m apps.vit.train config=apps/vit/configs/${dataset_name}.yaml ${run}"
            echo "Running command: ${command}"
            tmux send-keys -t ${session} "${command}" C-m
        done
    done
done


# CIFAR10-C
session="fin_cifar10_c_speckle"
dataset_name="cifar10_c"
corruption="speckle_noise"
batch_size=128
val_batch_size=128
grad_acc_steps=4
severity=5
device="cuda:0"

# Runs
tmux new-session -d -s ${session}
for seed in \
    0 \
    42 \
    3407
do
    for lr in \
        1e-3 \
        3e-3 \
        1e-2 \
        3e-2
    do
        for i in "${!comps[@]}"
        do
            log_dir="vit_${dataset_name}_${corruption}_${severity}_seed_${seed}_lr_${lr}_comp_${i}"
            run="log_dir=${log_dir} dataset_name=${dataset_name}-corruption-${corruption}-severity-${severity} seed=${seed} lr=${lr} ${comps[$i]} device=${device}"
            compute="batch_size=${batch_size} val_batch_size=${val_batch_size} grad_acc_steps=${grad_acc_steps}"
            command="python -m apps.vit.train config=apps/vit/configs/${dataset_name}.yaml ${run} ${compute}"
            echo "Running command: ${command}"
            tmux send-keys -t ${session} "${command}" C-m
        done
    done
done

# DOMAINNET
session="fin_domainnet_clipart"
dataset_name="domainnet"
domain="clipart"
device="cuda:0"

# Runs
tmux new-session -d -s ${session}
for seed in \
    0 \
    42 \
    3407
do
    for lr in \
        3e-3 \
        1e-2 \
        3e-2 \
        6e-2
    do
        for i in "${!comps[@]}"
        do
            log_dir="vit_${dataset_name}_${domain}_seed_${seed}_lr_${lr}_comp_${i}"
            run="log_dir=${log_dir} dataset_name=${dataset_name}-${domain} seed=${seed} lr=${lr} ${comps[$i]} device=${device}"
            command="python -m apps.vit.train config=apps/vit/configs/${dataset_name}.yaml ${run}"
            echo "Running command: ${command}"
            tmux send-keys -t ${session} "${command}" C-m
        done
    done
done

# DOMAINNET
session="fin_domainnet_sketch"
dataset_name="domainnet"
domain="sketch"
device="cuda:0"

# Runs
tmux new-session -d -s ${session}
for seed in \
    0 \
    42 \
    3407
do
    for lr in \
        3e-3 \
        1e-2 \
        3e-2 \
        6e-2
    do
        for i in "${!comps[@]}"
        do
            log_dir="vit_${dataset_name}_${domain}_seed_${seed}_lr_${lr}_comp_${i}"
            run="log_dir=${log_dir} dataset_name=${dataset_name}-${domain} seed=${seed} lr=${lr} ${comps[$i]} device=${device}"
            command="python -m apps.vit.train config=apps/vit/configs/${dataset_name}.yaml ${run}"
            echo "Running command: ${command}"
            tmux send-keys -t ${session} "${command}" C-m
        done
    done
done