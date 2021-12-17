import math


PI = 3.141592653589793
INIT_STEPS = 600
SCHEDULE_STEPS = 10000
NUM_CYCLES = 2
MIN_PERCENT = 1e-3


def anneal_fn(
    fn,
    current_step,
    num_init_steps=INIT_STEPS,
    num_anneal_steps=SCHEDULE_STEPS,
    num_cycles=NUM_CYCLES,
    value1=0.0,
    value2=1.0
):
    if current_step < num_init_steps:
        return anneal_fn(
            fn,
            current_step,
            num_init_steps=0,
            num_anneal_steps=num_init_steps * 2,
            num_cycles=1,
            value1=value2,
            value2=value1
        )
    if current_step > num_anneal_steps:
        return value2

    if not fn or fn == "none" or fn == "constant":
        return value2

    progress = float(num_cycles * (current_step - num_init_steps)) / max(1, num_anneal_steps - num_init_steps) % 1

    if fn == "linear":
        if progress < 0.5:
            return float(value1 + (value2 - value1) * progress * 2)
        else:
            return value2
    elif fn == "cosine":
        if progress < 0.5:
            return float(value1 + (value2 - value1) * (1 - math.cos(PI * progress * 2)) / 2)
        else:
            return value2
    else:
        raise NotImplementedError


class AnnealManner:
    def __init__(
        self,
        fn,
        current_step=0,
        num_init_steps=INIT_STEPS,
        num_anneal_steps=SCHEDULE_STEPS,
        num_cycles=NUM_CYCLES,
        value1=0.0,
        value2=1.0
    ):
        self.fn = fn
        self.num_init_steps = num_init_steps
        self.num_anneal_steps = num_anneal_steps
        self.num_cycles = num_cycles
        self.value1 = value1
        self.value2 = value2

        self._step_count = current_step

    def step(self, step=None):
        value = anneal_fn(
            self.fn,
            self._step_count,
            num_init_steps=self.num_init_steps,
            num_anneal_steps=self.num_anneal_steps,
            num_cycles=self.num_cycles,
            value1=self.value1,
            value2=self.value2
        )

        if step is not None:
            self._step_count = step
        else:
            self._step_count += 1

        return value
