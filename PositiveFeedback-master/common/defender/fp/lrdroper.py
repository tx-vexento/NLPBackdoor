import numpy as np
import matplotlib.pyplot as plt


def select_lr_droper(drop_method, max_steps, open_step, min_lr):
    if drop_method == "linear":
        return LinearLRDroper(drop_method, max_steps, open_step, min_lr)
    elif drop_method == "exp":
        return ExpLRDroper(drop_method, max_steps, open_step, min_lr)


class LRDroper:
    def __init__(self, drop_method, max_steps, open_step, min_lr):
        self.drop_method = drop_method
        self.max_steps = max_steps
        self.open_step = open_step
        self.min_lr = min_lr


class LinearLRDroper(LRDroper):
    def __init__(self, drop_method, max_steps, open_step, min_lr):
        super().__init__(drop_method, max_steps, open_step, min_lr)
        self.lambda_ = 1 / (max_steps - open_step)

    def get_lr(self, step):
        return 1 - self.lambda_ * (step - self.open_step) + self.min_lr


class ExpLRDroper(LRDroper):
    def __init__(self, drop_method, max_steps, open_step, min_lr):
        super().__init__(drop_method, max_steps, open_step, min_lr)
        self.lambda_ = -np.log(1e-2) / (max_steps - open_step)

    def get_lr(self, step):
        return np.exp(-self.lambda_ * (step - self.open_step)) + self.min_lr


class LogLRDroper(LRDroper):
    def __init__(self, drop_method, max_steps, open_step):
        super().__init__(drop_method, max_steps, open_step)
        self.lambda_ = 1 / np.log1p((max_steps - open_step) / (max_steps - open_step))

    def get_lr(self, step):
        if step < self.open_step:
            return 1.0

        return 1 / (
            1
            + self.lambda_
            * np.log1p((step - self.open_step) / (self.max_steps - self.open_step))
        )


if __name__ == "__main__":
    max_steps = 1000
    open_step = 200

    linear_lr_droper = LinearLRDroper("linear", max_steps, open_step, min_lr=0.5)
    exp_lr_droper = ExpLRDroper("exp", max_steps, open_step, min_lr=0.5)

    steps = np.arange(open_step, max_steps, 10)
    linear_lrs = [linear_lr_droper.get_lr(step) for step in steps]
    exp_lrs = [exp_lr_droper.get_lr(step) for step in steps]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, linear_lrs, label="Linear LR Droper", linestyle="-", marker="o")
    plt.plot(steps, exp_lrs, label="Exp LR Droper", linestyle="--", marker="x")

    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Droper Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig("lr_droper.png", dpi=300)
