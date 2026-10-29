# MIT License
#
# Copyright (c) 2021 Soohwan Kim and Sangchun Ha and Soyoung Cho
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
from omegaconf import DictConfig
from torch.optim import Optimizer

from . import register_scheduler, LearningRateSchedulerConfigs
from .lr_scheduler import LearningRateScheduler

logger = logging.getLogger(__name__)


@dataclass
class TriStageLRSchedulerConfigs(LearningRateSchedulerConfigs):
    scheduler_name: str = field(default="tri_stage", metadata={"help": "Name of learning rate scheduler."})
    init_lr: float = field(default=1e-7, metadata={"help": "Initial learning rate."})
    init_lr_scale: float = field(default=0.01, metadata={"help": "Initial learning rate scale."})
    final_lr_scale: float = field(default=0.01, metadata={"help": "Final learning rate scale"})
    phase_ratio: str = field(
        default="(0.1, 0.4, 0.5)",
        metadata={
            "help": "Automatically sets warmup/hold/decay steps to the ratio "
            "specified here from max_updates. the ratios must add up to 1.0"
        },
    )
    total_steps: int = field(default=400000, metadata={"help": "Total training steps."})
    extend_on_resume: bool = field(default=False, metadata={"help": "Extend a restored decay without resetting its learning rate."})


@register_scheduler("tri_stage", dataclass=TriStageLRSchedulerConfigs)
class TriStageLRScheduler(LearningRateScheduler):
    r"""
    Tri-Stage Learning Rate Scheduler. Implement the learning rate scheduler in "SpecAugment"

    Similar to inverse_squre_root scheduler, but tri_stage learning rate employs
    three stages LR scheduling:

        - warmup stage, starting from `lr` * `init_lr_scale`, linearly
          increased to `lr` in `warmup_steps` iterations
        - hold stage, after `warmup_steps`, keep the LR as `lr` for `hold_steps`
          iterations
        - decay stage, after hold stage, decay LR exponetially to
          `lr` * `final_lr_scale` in `decay_steps`;
          after that LR is keep as `final_lr_scale` * `lr`

    During warmup::
      init_lr = cfg.init_lr_scale * cfg.lr
      lrs = torch.linspace(init_lr, cfg.lr, cfg.warmup_steps)
      lr = lrs[update_num]

    During hold::
      lr = cfg.lr

    During decay::
      decay_factor = - math.log(cfg.final_lr_scale) / cfg.decay_steps
      lr = cfg.lr * exp(- (update_num - warmup_steps - decay_steps) * decay_factor)
    
    Updated it to be CosineAnneaLLR Scheduler

    After that::
      lr = cfg.lr * cfg.final_lr_scale

    Args:
        optimizer (Optimizer): wrapped optimizer.
        configs (DictConfig): configuration set.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        configs: DictConfig,
    ):
        super(TriStageLRScheduler, self).__init__(optimizer, configs.lr_scheduler.init_lr)

        self.phase_ratio = eval(configs.lr_scheduler.phase_ratio)
        self.total_steps = int(configs.lr_scheduler.total_steps)
        self.extend_on_resume = bool(configs.lr_scheduler.get("extend_on_resume", False))
        self.resumed_decay = None

        self.warmup_steps = int(configs.lr_scheduler.total_steps * self.phase_ratio[0])
        self.hold_steps = int(configs.lr_scheduler.total_steps * self.phase_ratio[1])
        self.decay_steps = int(configs.lr_scheduler.total_steps * self.phase_ratio[2])

        self.peak_lr = configs.lr_scheduler.lr
        self.init_lr = configs.lr_scheduler.init_lr_scale * configs.lr_scheduler.lr
        self.final_lr = configs.lr_scheduler.final_lr_scale * configs.lr_scheduler.lr

        self.warmup_rate = (self.peak_lr - self.init_lr) / self.warmup_steps if self.warmup_steps != 0 else 0
        self.decay_factor = -math.log(configs.lr_scheduler.final_lr_scale) / self.decay_steps
        self.update_step = 0
        self.lr = self.init_lr
        # The first optimizer update happens before Lightning calls step().
        self.set_lr(self.optimizer, self.init_lr)

    def load_state_dict(self, state_dict):
        """Optionally extend only the future decay after restoring a checkpoint.

        The optimizer keeps its saved LR for the first resumed update. The new
        cosine reaches the saved final LR on the requested final update. Its
        anchor is checkpointed, so later resumes do not restart the cosine.
        """
        requested_total = self.total_steps
        extend = self.extend_on_resume
        super().load_state_dict(state_dict)
        self.total_steps = int(state_dict.get(
            "total_steps", self.warmup_steps + self.hold_steps + self.decay_steps
        ))
        self.extend_on_resume = extend
        if not extend or requested_total == self.total_steps:
            return
        if requested_total < self.total_steps or requested_total <= self.update_step:
            raise ValueError("extend_on_resume requires a later training endpoint")
        if self.update_step < self.warmup_steps + self.hold_steps:
            raise ValueError("extend_on_resume requires a checkpoint in the decay stage or later")
        self.resumed_decay = {
            "start_step": self.update_step,
            "start_lr": self.lr,
            "end_step": requested_total,
        }
        self.total_steps = requested_total
        logger.info(
            "Extended restored LR decay: step %s, LR %.12g -> step %s, LR %.12g",
            self.update_step, self.lr, self.total_steps, self.final_lr,
        )

    def _decide_stage(self):
        if self.update_step < self.warmup_steps:
            return 0, self.update_step

        offset = self.warmup_steps

        if self.update_step < offset + self.hold_steps:
            return 1, self.update_step - offset

        offset += self.hold_steps

        if self.update_step <= offset + self.decay_steps:
            # decay stage
            return 2, self.update_step - offset

        offset += self.decay_steps

        return 3, self.update_step - offset

    def step(self, val_loss: Optional[torch.FloatTensor] = None):
        stage, steps_in_stage = self._decide_stage()

        if self.resumed_decay is not None:
            anchor = self.resumed_decay
            progress = min(1.0, max(0.0,
                (self.update_step + 1 - anchor["start_step"])
                / (anchor["end_step"] - anchor["start_step"])
            ))
            self.lr = self.final_lr + 0.5 * (anchor["start_lr"] - self.final_lr) * (
                1 + math.cos(progress * math.pi)
            )
        elif stage == 0:
            self.lr = self.init_lr + self.warmup_rate * steps_in_stage
        elif stage == 1:
            self.lr = self.peak_lr
        elif stage == 2:
            # self.lr = self.peak_lr * math.exp(-self.decay_factor * steps_in_stage)
            self.lr = self.final_lr + 0.5 * (self.peak_lr - self.final_lr) * (1 + math.cos(steps_in_stage / self.decay_steps * math.pi))
        elif stage == 3:
            self.lr = self.final_lr
        else:
            raise ValueError("Undefined stage")

        self.set_lr(self.optimizer, self.lr)
        self.update_step += 1

        return self.lr
