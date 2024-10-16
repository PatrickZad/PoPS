from torch.optim.lr_scheduler import _LRScheduler
import torch
from typing import List
import math


class EpochBasedWarmupCosineAnnealingLR(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))

    When last_epoch=-1, sets initial lr as lr.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        iters_per_epoch: int,
        delay_iters: int = 0,
        eta_min_lr: int = 0,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch=-1,
    ):
        self.max_iters = max_iters
        self.delay_iters = delay_iters
        self.eta_min_lr = eta_min_lr
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        assert (
            self.delay_iters >= self.warmup_iters
        ), "Scheduler delay iters must be larger than warmup iters"
        super(EpochBasedWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)
        self.max_sched_epoch = (max_iters - delay_iters) // iters_per_epoch

        self.iters_per_epoch = iters_per_epoch

    def get_lr(self) -> List[float]:
        if self.last_epoch <= self.warmup_iters:
            warmup_factor = _get_warmup_factor_at_iter(
                self.warmup_method,
                self.last_epoch,
                self.warmup_iters,
                self.warmup_factor,
            )
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        elif self.last_epoch <= self.delay_iters:
            return self.base_lrs

        else:
            iters_in_epoch = (self.last_epoch - self.delay_iters) % self.iters_per_epoch
            if iters_in_epoch == 0:
                next_iter_epoch = (
                    self.last_epoch - self.delay_iters
                ) // self.iters_per_epoch + 1
                return [
                    self.eta_min_lr
                    + (base_lr - self.eta_min_lr)
                    * (1 + math.cos(math.pi * next_iter_epoch / self.max_sched_epoch))
                    / 2
                    for base_lr in self.base_lrs
                ]
            else:
                return self.get_last_lr()


def _get_warmup_factor_at_iter(
    method: str, iter: int, warmup_iters: int, warmup_factor: float
) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See https://arxiv.org/abs/1706.02677 for more details.
    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).
    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))
