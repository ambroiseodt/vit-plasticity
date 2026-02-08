r"""
Optimizers and schedulers.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import math
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch.nn as nn
from torch.optim import SGD, AdamW, Optimizer, lr_scheduler

from .utils import build_with_type_check

# ------------------------------------------------------------------------------
# Optimizer
# ------------------------------------------------------------------------------


@dataclass
class OptimizerConfig:
    r"""
    Optimizer configuration.

    Notes
    -----
    Practitioners should refer to the technical report of the model to be trained to choose hyperparameters.
    If not available, recent studies in the literature could be use [1]_. For Adam [2]_, the default betas
    (0.9, 0.999) are widely used except for language modeling, where (0.9, 0.95) is more common (see [3]_
    and https://github.com/facebookresearch/lingua). Note that all hyperparameters can be easily tuned.

    References
    ----------
    .. [1] A. Orvieto and R. Gower. In Search of Adam's Secret Sauce. arXiv 2025
    .. [2] I. Loshchiilov and F. Hutter. Decoupled Weight Decay Regularization. In ICLR 2019
    .. [3] M. Cattaneo and B. Shigida. Tuning Adam(W): Default $\beta_2$ May Be Too Large. arXiv 2025
    """

    # Optimizer
    lr: float = 1e-3
    weight_decay: float = 0.0
    betas: tuple = (0.9, 0.999)
    momentum: float = 0.0


def build_optimizer(config: dict[str, Any], model: nn.Module) -> Optimizer:
    r"""
    Build optimizer.

    Parameters
    ----------
    config: dict
        Dictionary containing the configuration details.
    model: nn.Module
        Model to train.

    Returns
    -------
    optimizer: Optimizer
        Optimizer.
    """

    optimizer = config.pop("optimizer", "adamw")
    config_obj = build_with_type_check(OptimizerConfig, config)

    # Optimizer
    match optimizer.lower():
        case "adamw":
            optimizer = AdamW(
                model.parameters(),
                lr=config_obj.lr,
                betas=config_obj.betas,
                weight_decay=config_obj.weight_decay,
            )
        case "sgd":
            optimizer = SGD(
                model.parameters(),
                lr=config_obj.lr,
                weight_decay=config_obj.weight_decay,
                momentum=config_obj.momentum,
            )
        case _:
            raise ValueError(f"Unknown optimizer '{optimizer}'. Choose between 'adamw' and 'sgd'.")

    return optimizer


# ------------------------------------------------------------------------------
# Scheduler
# ------------------------------------------------------------------------------


@dataclass
class SchedulerConfig:
    r"""
    Config file for the scheduler.

    Notes
    -----
    Practitioners should refer to the technical report of the model to be trained to choose hyperparameters.
    In the literature, linear scheduler has been widely used for vision models training, while cosine scheduler
    is more common for language modeling. Note that all hyperparameters can be easily tuned.
    """

    # Scheduler
    warmup: int = 2000
    min_factor: float = 0
    cycle_length: float = 1.0
    decay_fraction: float = 0.1


def lr_constant(step: int) -> float:
    r"""
    Constant learning rate.

    Parameters
    ----------
    step: int
        Current optimization step.
    """
    return 1.0


def lr_linear(step: int, warmup: int, min_factor: float, n_steps: int) -> float:
    r"""
    Linear learning rate scheduler with warmup.

    The learning rate slowly increases during the warmup to reach the wanted lr value and then
    linearly decays until reaching a constant stage of value min_factor x the given lr.

    Parameters
    ----------
    step: int
        Current optimization step.
    warmup: int
        Number of warmup steps.
    min_factor: float
        Factor that determines the last learning rate stage.
    n_steps:
        Number of training steps.
    """

    # Warmup
    if step < warmup:
        lr = float(step) / warmup

    # Linear decay
    elif step <= n_steps:
        s = float(step - warmup) / (n_steps - warmup)
        lr = s * min_factor + (1 - s)

    # Constant stage
    else:
        lr = min_factor
    return lr


def lr_cosine(step: int, warmup: int, min_factor: float, n_steps: int) -> float:
    r"""
    Cosine learning rate scheduler with warmup.

    The learning rate slowly increases during the warmup to reach the wanted lr value and then
    evolves following a cosine function until reaching a constant stage of value min_factor x the given lr.

    Parameters
    ----------
    step: int
        Current optimization step.
    warmup: int
        Number of warmup steps.
    min_factor: float
        Factor that determines the last learning rate stage.
    n_steps:
        Number of training steps.
    """
    assert warmup != n_steps, "Warmup and steps should not be equal"

    # Warmup
    if step < warmup:
        lr = float(step) / warmup

    # Cosine evolution
    elif step <= n_steps:
        s = float(step - warmup) / (n_steps - warmup)
        lr = min_factor + 0.5 * (1 - min_factor) * (math.cos(math.pi * s) + 1)

    # Constant stage
    else:
        lr = min_factor
    return lr


def lr_wsd(
    step: int,
    warmup: int,
    min_factor: float,
    decay_fraction: float,
    cycle_length: float,
    n_steps: int,
) -> float:
    r"""
    Cosine learning rate scheduler with warmup.

    The learning rate slowly increases during the warmup to reach the wanted lr value and then
    evolves following a cosine function until reaching a constant stage of value min_factor x the given lr.

    Parameters
    ----------
    step: int
        Current optimization step.
    warmup: int
        Number of warmup steps.
    min_factor: float
        Factor that determines the last learning rate stage.
    decay_fraction: float
        Fraction of time spent decaying.
    cycle_Length: float
        Length of cycles.
    n_steps:
        Number of training steps.

    Notes
    -----
    This scheduler is tailored for language models training to avoid having to
    pre-determine a fixed compute budget, e.g., for cosine learning rate scheduler.
    Implementation inspired from https://github.com/facebookresearch/lingua.

    References
    ----------
    .. [4] K. Wen et al. Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape View.
           In ICLR 2025
    """

    cycle_num = step // int(n_steps * cycle_length) + 1
    curr_n_steps = int(n_steps * cycle_length) * cycle_num
    decay_length = int(curr_n_steps * decay_fraction)

    # Edge case
    if step == n_steps:
        cycle_num -= 1
        curr_n_steps = n_steps

    # Warmup
    if step < warmup:
        lr = float(step) / warmup

    # Evolution
    elif step <= curr_n_steps - decay_length:
        lr = 1.0
    elif step > curr_n_steps - decay_length and step <= curr_n_steps:
        step_in_decay = step - (curr_n_steps - decay_length)
        progress = step_in_decay / decay_length
        lr = 1 / (progress * (1 / min_factor) + (1 - progress))

    # Constant stage
    else:
        lr = min_factor

    return lr


def build_scheduler(config: dict[str, Any], optimizer: Optimizer, n_steps: int) -> lr_scheduler.LambdaLR:
    r"""
    Build scheduler.

    Parameters
    ----------
    optimizer: Optimizer
        Optimizer to schedule.
    config: OptimizerConfig
        Optimizer config.
    n_steps: int
        Number of training steps.

    Returns
    -------
    scheduler: lr_scheduler.LambdaLR
        Scheduler.
    """

    scheduler = config.pop("scheduler", "constant")
    config_obj = build_with_type_check(SchedulerConfig, config)

    match scheduler.lower():
        case "constant":
            lr_lambda = lr_constant
        case "linear":
            lr_lambda = partial(
                lr_linear,
                warmup=config_obj.warmup,
                min_factor=config_obj.min_factor,
                n_steps=n_steps,
            )
        case "cosine":
            lr_lambda = partial(
                lr_cosine,
                warmup=config_obj.warmup,
                min_factor=config_obj.min_factor,
                n_steps=n_steps,
            )
        case "wsd":
            lr_lambda = partial(
                lr_wsd,
                warmup=config_obj.warmup,
                min_factor=config_obj.min_factor,
                decay_fraction=config_obj.decay_fraction,
                cycle_length=config_obj.cycle_length,
                n_steps=n_steps,
            )
        case _:
            raise ValueError(
                f"Unknown scheduler '{scheduler}'. Choose between 'constant', 'linear', 'cosine' and 'wsd'."
            )

    # Scheduler
    scheduler = lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)

    return scheduler
