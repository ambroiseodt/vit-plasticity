r"""
Utility functions for ViT training and evaluation.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from dataclasses import dataclass

import torch.nn as nn
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import lr_scheduler

# ------------------------------------------------------------------------------
# Monitor training state
# ------------------------------------------------------------------------------


@dataclass
class TrainingState(Stateful):
    r"""Training state config."""

    # Number of steps taken by the optimizer
    step: int

    # Number of accumulation steps done since last optimizer step
    acc_step: int

    # Scheduler
    scheduler: lr_scheduler.LambdaLR

    def state_dict(self) -> dict[str, int]:
        return {
            "step": self.step,
            "acc_step": self.acc_step,
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.step = state_dict["step"]
        self.acc_step = state_dict["acc_step"]
        self.scheduler.load_state_dict(state_dict["scheduler"])


# ------------------------------------------------------------------------------
# Freeze trainable weights
# ------------------------------------------------------------------------------


def freeze_model(model: nn.Module, components: list[str]) -> None:
    r"""
    Freeze the specified components of a ViT model across layers.

    Parameters:
    -----------
    model: nn.Module
        Model whose components we want to freeze.
    components: list[str]
        Elements of the list can be "emb", "attn_norm", "mha", "ffn_norm", "ffn_fc1", "ffn_activation" or "ffn_fc2".
    """

    # Mapping between components and ViT weights
    map_weight = {
        "emb": "embedding",
        "attn_norm": "attn_norm",
        "mha": ["attn.qkv_mat", "attn.output"],
        "ffn_norm": "ffn_norm",
        "ffn_fc1": "ffn.fc1",
        "ffn_fc2": "ffn.fc2",
    }

    # Recover list of weights to freeze
    weights = []
    for comp in components:
        weights.extend(map_weight[comp] if isinstance(map_weight[comp], list) else [map_weight[comp]])

    # Freeze embedding layer
    if "embedding" in weights:
        for param in model.model.embedding.parameters():
            param.requires_grad = False

    # Freeze transformer components
    blocks = model.model.blocks if hasattr(model, "model") else model.blocks
    for block in blocks:
        for name, param in block.named_parameters():
            if any(weight in name for weight in weights):
                param.requires_grad = False


# ------------------------------------------------------------------------------
# Evaluation State
# ------------------------------------------------------------------------------


@dataclass
class EvalState(Stateful):
    r"""Evaluation state config."""

    # Accuracy
    accuracy: float = 0

    # Loss
    loss: float = 0

    # Number of state in the evaluation loader
    step: int = 0

    def state_dict(self) -> dict:
        return {"eval_acc": self.accuracy, "eval_loss": self.loss, "step": self.step}

    def load_state_dict(self, state_dict: dict) -> None:
        self.accuracy = state_dict["eval_acc"]
        self.loss = state_dict["eval_loss"]
        self.step = state_dict["step"]
