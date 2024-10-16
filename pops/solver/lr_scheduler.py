# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import List
import torch
from fvcore.common.param_scheduler import (
    CompositeParamScheduler,
    ConstantParamScheduler,
    LinearParamScheduler,
    ParamScheduler,
)
import math

logger = logging.getLogger(__name__)


class WarmupParamScheduler(CompositeParamScheduler):
    """
    Add an initial warmup stage to another scheduler.
    """

    def __init__(
        self,
        scheduler: ParamScheduler,
        warmup_factor: float,
        warmup_length: float,
        warmup_method: str = "linear",
    ):
        """
        Args:
            scheduler: warmup will be added at the beginning of this scheduler
            warmup_factor: the factor w.r.t the initial value of ``scheduler``, e.g. 0.001
            warmup_length: the relative length (in [0, 1]) of warmup steps w.r.t the entire
                training, e.g. 0.01
            warmup_method: one of "linear" or "constant"
        """
        end_value = scheduler(warmup_length)  # the value to reach when warmup ends
        start_value = warmup_factor * scheduler(0.0)
        if warmup_method == "constant":
            warmup = ConstantParamScheduler(start_value)
        elif warmup_method == "linear":
            warmup = LinearParamScheduler(start_value, end_value)
        else:
            raise ValueError("Unknown warmup method: {}".format(warmup_method))
        super().__init__(
            [warmup, scheduler],
            interval_scaling=["rescaled", "fixed"],
            lengths=[warmup_length, 1 - warmup_length],
        )


class LRMultiplier(torch.optim.lr_scheduler._LRScheduler):
    """
    A LRScheduler which uses fvcore :class:`ParamScheduler` to multiply the
    learning rate of each param in the optimizer.
    Every step, the learning rate of each parameter becomes its initial value
    multiplied by the output of the given :class:`ParamScheduler`.

    The absolute learning rate value of each parameter can be different.
    This scheduler can be used as long as the relative scale among them do
    not change during training.

    Examples:
    ::
        LRMultiplier(
            opt,
            WarmupParamScheduler(
                MultiStepParamScheduler(
                    [1, 0.1, 0.01],
                    milestones=[60000, 80000],
                    num_updates=90000,
                ), 0.001, 100 / 90000
            ),
            max_iter=90000
        )
    """

    # NOTES: in the most general case, every LR can use its own scheduler.
    # Supporting this requires interaction with the optimizer when its parameter
    # group is initialized. For example, classyvision implements its own optimizer
    # that allows different schedulers for every parameter group.
    # To avoid this complexity, we use this class to support the most common cases
    # where the relative scale among all LRs stay unchanged during training.  In this
    # case we only need a total of one scheduler that defines the relative LR multiplier.

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        multiplier: ParamScheduler,
        max_iter: int,
        last_iter: int = -1,
    ):
        """
        Args:
            optimizer, last_iter: See ``torch.optim.lr_scheduler._LRScheduler``.
                ``last_iter`` is the same as ``last_epoch``.
            multiplier: a fvcore ParamScheduler that defines the multiplier on
                every LR of the optimizer
            max_iter: the total number of training iterations
        """
        if not isinstance(multiplier, ParamScheduler):
            raise ValueError(
                "_LRMultiplier(multiplier=) must be an instance of fvcore "
                f"ParamScheduler. Got {multiplier} instead."
            )
        self._multiplier = multiplier
        self._max_iter = max_iter
        super().__init__(optimizer, last_epoch=last_iter)

    def state_dict(self):
        # fvcore schedulers are stateless. Only keep pytorch scheduler states
        return {"base_lrs": self.base_lrs, "last_epoch": self.last_epoch}

    def get_lr(self) -> List[float]:
        multiplier = self._multiplier(self.last_epoch / self._max_iter)
        return [base_lr * multiplier for base_lr in self.base_lrs]


class EpochBasedCosineParamScheduler(ParamScheduler):
    def __init__(
        self,
        start_value: float,
        end_value: float,
        epoch_base: int,
        max_iters: int,
        warmup_iters: int = 0,
    ) -> None:
        self._start_value = start_value
        self._end_value = end_value
        self.epoch_base = epoch_base
        self.max_iters = max_iters
        self.max_epoch_after_warmup = math.ceil(
            (self.max_iters - warmup_iters) / epoch_base
        )
        self.warmup_iters = warmup_iters

    def __call__(self, where: float) -> float:
        cur_iter = where * self.max_iters - self.warmup_iters
        cur_epoch = cur_iter // self.epoch_base
        where = cur_epoch / self.max_epoch_after_warmup
        return self._end_value + 0.5 * (self._start_value - self._end_value) * (
            1 + math.cos(math.pi * where)
        )


class CosineAfterWarmupParamScheduler(ParamScheduler):
    """
    Cosine decay or cosine warmup schedules based on start and end values.
    The schedule is updated based on the fraction of training progress.
    The schedule was proposed in 'SGDR: Stochastic Gradient Descent with
    Warm Restarts' (https://arxiv.org/abs/1608.03983). Note that this class
    only implements the cosine annealing part of SGDR, and not the restarts.

    Example:

        .. code-block:: python

          CosineParamScheduler(start_value=0.1, end_value=0.0001)
    """

    def __init__(
        self,
        start_value: float,
        end_value: float,
        max_iters: int,
        warmup_iters: int = 0,
    ) -> None:
        self._start_value = start_value
        self._end_value = end_value
        self.max_iters = max_iters
        self.warmup_iters = warmup_iters

    def __call__(self, where: float) -> float:
        cur_iter = where * self.max_iters - self.warmup_iters
        where = cur_iter / (self.max_iters - self.warmup_iters)
        return self._end_value + 0.5 * (self._start_value - self._end_value) * (
            1 + math.cos(math.pi * where)
        )
