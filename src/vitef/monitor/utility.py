r"""
Utility manager for seed setting and garbage collection.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import gc
import logging
from dataclasses import dataclass
from types import TracebackType

from ..config import set_seed

logger = logging.getLogger("vitef")


@dataclass
class UtilityConfig:
    r"""
    Utility configuration.

    Parameters
    ----------
    seed: int
        Seed for reproducibility.
    period: int
        Number of updates between each garbage collection.
    """

    seed: int = 42
    period: int = 1000


class Utility:
    def __init__(self, config: UtilityConfig):
        self.seed = config.seed
        self.period = config.period
        self.step = 0

    def __enter__(self) -> "Utility":
        r"""Enter the utility manager to control garbage collection."""

        # Reproducibility
        set_seed(self.seed)

        # Disable automatic garbage collection
        gc.disable()

        # Collect garbage
        gc.collect()
        return self

    def __call__(self) -> None:
        r"""Run garbage collection periodically."""
        self.step += 1
        if self.period <= 0:
            return
        if self.step % self.period == 0:
            logger.info("garbage collection")
            gc.collect()

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType) -> None:
        r"""Exit the utility manager."""
        # Enable automatic garbage collection."""
        gc.enable()
        return
