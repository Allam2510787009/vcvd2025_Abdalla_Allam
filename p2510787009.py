import argparse
import numpy as np

from vehicle import Vehicle
from tyre_model import TyreModel
from plotter import Plotter


def main():
    parser = argparse.ArgumentParser(description="Pacejka tyre model simulation")
    parser.add_argument("--alpha", required=True,
                        help="Slip angle(s) in degrees, comma separated (e.g. 2,5,8)")
    parser.add_argument("--weight", required=True, type=float,
                        help="Vehicle mass in kg")
    parser.add_argument("--gamma", default=0.0, type=float,
                        help="Camber angle in degrees (default 0)")
    args = parser.parse_args()

    # Parse inputs
    slip_angles_deg = [float(a) for a in args.alpha.split(",")]
    slip_angles_rad = [np.deg2rad(a) for a in slip_angles_deg]
    gamma_rad = np.deg2rad(args.gamma)

    # Create objects
    vehicle = Vehicle(args.weight)
    tyre = TyreModel("coeffs.json")
    plotter = Plotter()

    # Vertical load per wheel
    Fz = vehicle.wheel_load()

    # Longitudinal slip range (0–100%)
    kappa = np.linspace(0.0, 1.0, 200)

    fx_results = {}
    fy_results = {}

    for alpha_deg, alpha_rad in zip(slip_angles_deg, slip_angles_rad):
        fx = tyre.longitudinal_force(kappa, Fz)
        fy = tyre.lateral_force(alpha_rad, kappa, Fz, gamma_rad)

        fx_results[alpha_deg] = fx
        fy_results[alpha_deg] = fy

    # Plot results
    plotter.plot_longitudinal_force(kappa, fx_results)
    plotter.plot_lateral_force(kappa, fy_results)


if __name__ == "__main__":
    main()
