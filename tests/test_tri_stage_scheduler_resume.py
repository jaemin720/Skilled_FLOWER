"""Check LR continuity and checkpoint persistence without using a GPU."""

import copy
import unittest

from omegaconf import OmegaConf
import torch

from flower.utils.lr_schedulers.tri_stage_scheduler import TriStageLRScheduler


def make_scheduler(total_steps=100, extend=False):
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=2e-5)
    config = OmegaConf.create({"lr_scheduler": {
        "init_lr": 2e-5, "lr": 2e-5, "init_lr_scale": 0.1,
        "final_lr_scale": 0.5, "phase_ratio": "(0.05, 0.1, 0.85)",
        "total_steps": total_steps, "extend_on_resume": extend,
    }})
    return optimizer, TriStageLRScheduler(optimizer, config)


def old_checkpoint():
    optimizer, scheduler = make_scheduler()
    for _ in range(25):
        scheduler.step()
    state = copy.deepcopy(scheduler.state_dict())
    for key in ("total_steps", "extend_on_resume", "resumed_decay"):
        state.pop(key)  # The five-epoch checkpoint predates these fields.
    return copy.deepcopy(optimizer.state_dict()), state


class ResumeSchedulerTests(unittest.TestCase):
    def test_existing_resume_keeps_original_schedule(self):
        optimizer_state, scheduler_state = old_checkpoint()
        o1, s1 = make_scheduler()
        o2, s2 = make_scheduler(200)
        for optimizer, scheduler in [(o1, s1), (o2, s2)]:
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        self.assertIsNone(s2.resumed_decay)
        for _ in range(100):
            self.assertEqual(s1.step(), s2.step())

    def test_extension_preserves_lr_and_reaches_new_endpoint(self):
        optimizer_state, scheduler_state = old_checkpoint()
        original = copy.deepcopy(scheduler_state)
        optimizer, scheduler = make_scheduler(200, extend=True)
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(scheduler_state)
        previous_lr = optimizer.param_groups[0]["lr"]
        self.assertEqual(previous_lr, scheduler_state["lr"])
        self.assertEqual(scheduler.update_step, 25)
        self.assertEqual(scheduler_state, original)
        for _ in range(175):
            current_lr = scheduler.step()
            self.assertLessEqual(current_lr, previous_lr)
            self.assertGreaterEqual(current_lr, 1e-5)
            previous_lr = current_lr
        self.assertEqual(scheduler.update_step, 200)
        self.assertEqual(current_lr, 1e-5)
        self.assertEqual(scheduler.step(), 1e-5)

    def test_resuming_extended_schedule_keeps_anchor_and_future_lrs(self):
        optimizer_state, scheduler_state = old_checkpoint()
        optimizer, scheduler = make_scheduler(200, extend=True)
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(scheduler_state)
        for _ in range(30):
            scheduler.step()
        other_optimizer, other_scheduler = make_scheduler(200, extend=True)
        other_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        other_scheduler.load_state_dict(copy.deepcopy(scheduler.state_dict()))
        self.assertEqual(other_scheduler.resumed_decay["start_step"], 25)
        for _ in range(150):
            self.assertEqual(scheduler.step(), other_scheduler.step())

    def test_shortening_and_warmup_extension_are_rejected(self):
        _, state = old_checkpoint()
        _, shorter = make_scheduler(90, extend=True)
        with self.assertRaisesRegex(ValueError, "later training endpoint"):
            shorter.load_state_dict(state)
        _, early = make_scheduler()
        early.step()
        _, longer = make_scheduler(200, extend=True)
        with self.assertRaisesRegex(ValueError, "decay stage"):
            longer.load_state_dict(early.state_dict())


if __name__ == "__main__":
    unittest.main()
