import math
from torch.optim.lr_scheduler import LambdaLR


PI = 3.141592653589793
INIT_STEPS = 600
SCHEDULE_STEPS = 10000
NUM_CYCLES = 2
MIN_PERCENT = 1e-3


class ConstantScheduler(LambdaLR):
    def __init__(self):
        pass

    def set_optimizer(self, optimizer):
        super(ConstantScheduler, self).__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, current_step):
        return 1.0


class ConstantWarmupScheduler(LambdaLR):
    def __init__(
        self,
        num_warmup_steps=INIT_STEPS
    ):
        self.num_warmup_steps = num_warmup_steps

    def set_optimizer(self, optimizer):
        super(ConstantWarmupScheduler, self).__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, current_step):
        if current_step < self.num_warmup_steps:
            return float(current_step) / max(1.0, float(self.num_warmup_steps))
        return 1.0


class LinearScheduler(LambdaLR):
    def __init__(
        self,
        num_schedule_steps=SCHEDULE_STEPS,
        min_percent=MIN_PERCENT
    ):
        self.num_schedule_steps = num_schedule_steps
        self.min_percent = min_percent

    def set_optimizer(self, optimizer):
        super(LinearScheduler, self).__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, current_step):
        return max(
            self.min_percent,
            float(self.num_schedule_steps - current_step) / \
                float(max(1, self.num_schedule_steps))
        )


class LinearWarmupScheduler(LambdaLR):
    def __init__(
        self,
        num_warmup_steps=INIT_STEPS,
        num_schedule_steps=SCHEDULE_STEPS,
        min_percent=MIN_PERCENT
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_schedule_steps = num_schedule_steps
        self.min_percent = min_percent

    def set_optimizer(self, optimizer):
        super(LinearWarmupScheduler, self).__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, current_step):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        return max(
            self.min_percent,
            float(self.num_schedule_steps - current_step) / \
                float(max(1, self.num_schedule_steps - self.num_warmup_steps))
        )


class LinearWarmupRestartScheduler(LambdaLR):
    def __init__(
        self,
        num_warmup_steps=INIT_STEPS,
        num_schedule_steps=SCHEDULE_STEPS,
        num_cycles=NUM_CYCLES,
        min_percent=MIN_PERCENT
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_schedule_steps = num_schedule_steps
        self.num_cycles = num_cycles
        self.min_percent = min_percent

    def set_optimizer(self, optimizer):
        super(LinearWarmupRestartScheduler, self).__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, current_step):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        progress = float(current_step - self.num_warmup_steps) / \
            float(max(1, self.num_schedule_steps - self.num_warmup_steps))
        if progress >= 1.0:
            return self.min_percent
        return max(self.min_percent, 1 - (float(self.num_cycles) * progress) % 1.0)


class CosineScheduler(LambdaLR):
    def __init__(
        self,
        num_schedule_steps=SCHEDULE_STEPS,
        num_cycles=NUM_CYCLES,
        min_percent=MIN_PERCENT
    ):
        self.num_schedule_steps = num_schedule_steps
        self.num_cycles = num_cycles
        self.min_percent = min_percent

    def set_optimizer(self, optimizer):
        super(CosineScheduler, self).__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, current_step):
        progress = float(current_step) / float(max(1, self.num_schedule_steps))
        return max(self.min_percent, 0.5 * (1.0 + math.cos(PI * float(self.num_cycles) * 2.0 * progress)))


class CosineWarmupScheduler(LambdaLR):
    def __init__(
        self,
        num_warmup_steps=INIT_STEPS,
        num_schedule_steps=SCHEDULE_STEPS,
        num_cycles=NUM_CYCLES,
        min_percent=MIN_PERCENT
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_schedule_steps = num_schedule_steps
        self.num_cycles = num_cycles
        self.min_percent = min_percent

    def set_optimizer(self, optimizer):
        super(CosineWarmupScheduler, self).__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, current_step):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        progress = float(current_step - self.num_warmup_steps) / \
            float(max(1, self.num_schedule_steps - self.num_warmup_steps))
        return max(self.min_percent, 0.5 * (1.0 + math.cos(PI * float(self.num_cycles) * 2.0 * progress)))


class CosineWarmupRestartScheduler(LambdaLR):
    def __init__(
        self,
        num_warmup_steps=INIT_STEPS,
        num_schedule_steps=SCHEDULE_STEPS,
        num_cycles=NUM_CYCLES,
        min_percent=MIN_PERCENT
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_schedule_steps = num_schedule_steps
        self.num_cycles = num_cycles
        self.min_percent = min_percent

    def set_optimizer(self, optimizer):
        super(CosineWarmupRestartScheduler, self).__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, current_step):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        progress = float(current_step - self.num_warmup_steps) / \
            float(max(1, self.num_schedule_steps - self.num_warmup_steps))
        if progress >= 1.0:
            return self.min_percent
        return max(self.min_percent, 0.5 * (1.0 + math.cos(PI * ((float(self.num_cycles) * progress) % 1.0))))


supported_schedulers = {
    "constant": ConstantScheduler(),
    "constant_with_warmup": ConstantWarmupScheduler(),
    "linear": LinearScheduler(),
    "linear_with_warmup": LinearWarmupScheduler(),
    "linear_with_warmup_and_restart": LinearWarmupRestartScheduler(),
    "cosine": CosineScheduler(),
    "cosine_with_warmup": CosineWarmupScheduler(),
    "cosine_with_warmup_and_restart": CosineWarmupRestartScheduler(),
}


def map_scheduler_str_to_scheduler(scheduler, **kw):
    if scheduler not in supported_schedulers:
        raise NotImplementedError

    sdlr = supported_schedulers[scheduler]
    for k, v in kw.items():
        if hasattr(sdlr, k):
            try:
                setattr(sdlr, k, v)
            except:
                pass
    return sdlr
