import os
import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    """
    Handles plotting and saving of tyre force characteristics.
    """

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_longitudinal_force(self, slip: np.ndarray, fx_dict: dict):
        """
        Plot longitudinal force vs longitudinal slip.

        fx_dict: { slip_angle_deg : Fx_array }
        """
        plt.figure()
        for alpha, fx in fx_dict.items():
            plt.plot(slip * 100.0, fx, label=f"α = {alpha}°")

        plt.xlabel("Longitudinal slip (%)")
        plt.ylabel("Longitudinal force Fx (N)")
        plt.title("Longitudinal Force vs Longitudinal Slip")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "longitudinal_force.png"), dpi=150)
        plt.close()

    def plot_lateral_force(self, slip: np.ndarray, fy_dict: dict):
        """
        Plot lateral force vs longitudinal slip.

        fy_dict: { slip_angle_deg : Fy_array }
        """
        plt.figure()
        for alpha, fy in fy_dict.items():
            plt.plot(slip * 100.0, fy, label=f"α = {alpha}°")

        plt.xlabel("Longitudinal slip (%)")
        plt.ylabel("Lateral force Fy (N)")
        plt.title("Lateral Force vs Longitudinal Slip")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "lateral_force.png"), dpi=150)
        plt.close()
