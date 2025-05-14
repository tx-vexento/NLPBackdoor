import matplotlib.pyplot as plt
import os
import numpy as np
import json


class Smoother:
    def __init__(self, window_size=200):
        self.window_size = window_size
        self.values = []
        self.presum_values = []
        self.moving_averaged_values = []
        self.min_value_eps = 1e-2

    def update(self, new_value):
        self.values.append(new_value)
        if len(self.presum_values) == 0:
            self.presum_values.append(new_value)
        else:
            self.presum_values.append(self.presum_values[-1] + new_value)
        moving_averaged_value = self.calculate_moving_average()
        self.moving_averaged_values.append(moving_averaged_value)

        if moving_averaged_value < self.min_value_eps:
            return True
        return False

    def calculate_moving_average(self):
        if len(self.values) < self.window_size:
            return sum(self.values) / len(self.values)
        else:
            return sum(self.values[-self.window_size :]) / self.window_size

    def plot(self, output_dir):
        plt.figure(figsize=(10, 6))
        plt.scatter(
            list(range(len(self.values))), self.values, label="similarity", marker="o"
        )
        plt.plot(self.moving_averaged_values, label="moving average", marker="x")
        # plt.plot(self.presum_values, label="presum", marker="s")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.title("Moving Average Smoothing")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "moving_average_smoothing.png"), dpi=300)

        lowup_output_dir = os.path.join(output_dir, "lowup")
        with open(os.path.join(lowup_output_dir, "lowup-grad-sim.jsonl"), "w") as f:
            f.writelines(
                [
                    json.dumps(
                        {
                            "val": self.values[i],
                            "smooth": self.moving_averaged_values[i],
                        }
                    )
                    + "\n"
                    for i in range(len(self.values))
                ]
            )
