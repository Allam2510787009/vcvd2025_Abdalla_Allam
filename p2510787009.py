"""
Pacejka Magic Formula Tyre Model
=================================
Implements the tyre model as described in:

[1] Bakker, Egbert and Nyborg, Lars and Pacejka, Hans B.,
    "Tyre Modelling for Use in Vehicle Dynamics Studies",
    SAE Transactions, 1987

Produces plots of side force Fy and brake force Fx vs longitudinal
slip kappa (0-100%), with normal load Fz as the curve parameter, at
a fixed slip angle alpha. This matches Figs. 19 and 20 in [1].

Usage example:
    python 2510787009.py --slip 2 --weight 1500 --mu 0.8
    python 2510787009.py --slip 5 --weight 1500 --mu 0.8
"""

# ==========================================================
# Standard library and third-party imports
# ==========================================================
import argparse
import json
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.constants import g as GRAVITY_MS2

# ==========================================================
# Module-level constants
# ==========================================================
SLIP_RANGE_START = 0.0
SLIP_RANGE_END = 100.0
SLIP_STEPS = 500
NUM_WHEELS = 4
FZ_KN_CONVERSION = 1000.0
PERCENT_TO_FRACTION = 0.01
DEG_TO_RAD = math.pi / 180.0

COEFFICIENTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "tyre_coefficients.json"
)

# Fixed Fz levels shown in Figs 19/20 of [1], in Newtons
FZ_LEVELS_NEWTON = [8000.0, 6000.0, 4000.0, 2000.0]


# ==========================================================
# Data loading
# ==========================================================
def load_coefficients(filepath):
    """Load tyre formula coefficients from a JSON file.

    param in: filepath ... path to the JSON coefficients file
    returns: tuple (coeffs_fy, coeffs_fx, coeffs_fy_camber)
    """
    with open(filepath, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    coeffs_fy = data["table2"]["side_force_Fy"]
    coeffs_fx = data["table2"]["brake_force_Fx"]
    coeffs_fy_camber = data["table3"]["side_force_Fy"]
    return coeffs_fy, coeffs_fx, coeffs_fy_camber


# ==========================================================
# Vehicle load
# ==========================================================
def calc_wheel_load(vehicle_weight_kg):
    """Derive normal load per wheel assuming equal distribution over 4 wheels.

    param in: vehicle_weight_kg ... total vehicle mass in kg
    returns: normal force Fz in N per wheel
    """
    total_force_newton = vehicle_weight_kg * GRAVITY_MS2
    fz_per_wheel = total_force_newton / NUM_WHEELS
    return fz_per_wheel


# ==========================================================
# Magic Formula – Pure Brake Force Fx
# ===========================================-==============
def calc_fx_pure(slip_kappa_percent, fz_newton, coeffs_fx):
    """Compute pure longitudinal brake force Fx using Magic Formula [1].

    param in: slip_kappa_percent ... longitudinal slip kappa in percent (0-100)
    param in: fz_newton          ... normal load in N
    param in: coeffs_fx          ... dict of a1..a8 for Fx
    returns: pure brake force Fx in N
    """
    fz_kn = fz_newton / FZ_KN_CONVERSION
    kappa = slip_kappa_percent * PERCENT_TO_FRACTION

    coeff_a1 = coeffs_fx["a1"]
    coeff_a2 = coeffs_fx["a2"]
    coeff_a3 = coeffs_fx["a3"]
    coeff_a4 = coeffs_fx["a4"]
    coeff_a5 = coeffs_fx["a5"]
    coeff_a6 = coeffs_fx["a6"]
    coeff_a7 = coeffs_fx["a7"]
    coeff_a8 = coeffs_fx["a8"]

    peak_d = coeff_a1 * fz_kn ** 2 + coeff_a2 * fz_kn
    shape_c = 1.65
    stiffness_bcd = coeff_a3 * fz_kn ** 2 + coeff_a4 * fz_kn
    curvature_e = coeff_a6 * fz_kn ** 2 + coeff_a7 * fz_kn + coeff_a8

    stiffness_b = stiffness_bcd / (
        shape_c * peak_d * math.exp(coeff_a5 * fz_kn) + 1e-9
    )
    phi_val = (1.0 - curvature_e) * kappa + (
        curvature_e / stiffness_b
    ) * math.atan(stiffness_b * kappa)

    fx_pure = peak_d * math.sin(shape_c * math.atan(stiffness_b * phi_val))
    return fx_pure


# ============================================================
# Magic Formula – Pure Side Force Fy
# ==========================================================
def calc_fy_pure(slip_angle_deg, fz_newton, camber_deg, coeffs_fy, coeffs_fy_camber):
    """Compute pure lateral side force Fy using Magic Formula [1].

    param in: slip_angle_deg     ... slip angle alpha in degrees
    param in: fz_newton          ... normal load in N
    param in: camber_deg         ... camber angle gamma in degrees
    param in: coeffs_fy          ... dict of a1..a8 for Fy
    param in: coeffs_fy_camber   ... dict of a9..a13 for Fy camber
    returns: pure side force Fy in N
    """
    fz_kn = fz_newton / FZ_KN_CONVERSION
    alpha = slip_angle_deg
    gamma = camber_deg

    coeff_a1  = coeffs_fy["a1"]
    coeff_a2  = coeffs_fy["a2"]
    coeff_a3  = coeffs_fy["a3"]
    coeff_a4  = coeffs_fy["a4"]
    coeff_a5  = coeffs_fy["a5"]
    coeff_a6  = coeffs_fy["a6"]
    coeff_a7  = coeffs_fy["a7"]
    coeff_a8  = coeffs_fy["a8"]
    coeff_a9  = coeffs_fy_camber["a9"]
    coeff_a10 = coeffs_fy_camber["a10"]
    coeff_a11 = coeffs_fy_camber["a11"]
    coeff_a12 = coeffs_fy_camber["a12"]

    peak_d = coeff_a1 * fz_kn ** 2 + coeff_a2 * fz_kn
    shape_c = 1.30
    stiffness_bcd = (
        coeff_a3 * math.sin(2.0 * math.atan(fz_kn / coeff_a4))
        * (1.0 - coeff_a5 * abs(gamma))
    )
    curvature_e = coeff_a6 * fz_kn ** 2 + coeff_a7 * fz_kn + coeff_a8
    sh_offset = coeff_a9 * gamma
    sv_offset = (coeff_a10 * fz_kn ** 2 + coeff_a11 * fz_kn) * gamma

    stiffness_b = stiffness_bcd / (shape_c * peak_d + 1e-9)
    phi_val = (1.0 - curvature_e) * (alpha + sh_offset) + (
        curvature_e / stiffness_b
    ) * math.atan(stiffness_b * (alpha + sh_offset))

    fy_pure = peak_d * math.sin(shape_c * math.atan(stiffness_b * phi_val)) + sv_offset
    return fy_pure


# ==========================================================
# Combined slip: Fx and Fy under simultaneous alpha and kappa
# ==================================[=======================
def calc_combined_forces(slip_kappa_percent, slip_angle_deg, fz_newton,
                         coeffs_fy, coeffs_fx, coeffs_fy_camber):
    """Compute Fx and Fy under combined longitudinal and lateral slip [1].

    Uses the resultant-slip method: the combined slip is computed from
    kappa and tan(alpha), then pure-slip Magic Formula forces are scaled
    proportionally to give Fx and Fy components.

    param in: slip_kappa_percent ... longitudinal slip kappa in %
    param in: slip_angle_deg     ... slip angle alpha in degrees
    param in: fz_newton          ... normal load in N
    param in: coeffs_fy          ... Fy coefficient dict
    param in: coeffs_fx          ... Fx coefficient dict
    param in: coeffs_fy_camber   ... Fy camber coefficient dict
    returns: tuple (fx_combined, fy_combined) in N
    """
    kappa = slip_kappa_percent * PERCENT_TO_FRACTION
    tan_alpha = math.tan(slip_angle_deg * DEG_TO_RAD)
    sigma_combined = math.sqrt(kappa ** 2 + tan_alpha ** 2)

    if sigma_combined < 1e-9:
        fy_val = calc_fy_pure(slip_angle_deg, fz_newton, 0.0,
                              coeffs_fy, coeffs_fy_camber)
        return 0.0, abs(fy_val)

    # Evaluate pure forces at the combined slip magnitude
    sigma_pct = sigma_combined / PERCENT_TO_FRACTION
    fx0 = calc_fx_pure(sigma_pct, fz_newton, coeffs_fx)
    alpha_equiv = math.degrees(math.atan(sigma_combined))
    fy0 = calc_fy_pure(alpha_equiv, fz_newton, 0.0, coeffs_fy, coeffs_fy_camber)

    # Resultant force
    force_resultant = math.sqrt(fx0 ** 2 + fy0 ** 2)
    if force_resultant < 1e-9:
        return 0.0, 0.0

    # Project onto kappa and tan_alpha axes
    fx_combined = force_resultant * kappa / sigma_combined
    fy_combined = force_resultant * abs(tan_alpha) / sigma_combined
    return fx_combined, fy_combined


# ==========================================================
# Build force curve arrays for all Fz levels
# ==========================================================
def build_force_curves(slip_angle_deg, fz_levels_newton,
                       coeffs_fy, coeffs_fx, coeffs_fy_camber):
    """Build Fx and Fy arrays vs kappa for each Fz level.

    parameter in: slip_angle_deg      ... fixed slip angle alpha in degrees
    parameter in: fz_levels_newton    ... list of normal loads in N
    parameter in: coeffs_fy           ... Fy coefficient dict
    parameter in: coeffs_fx           ... Fx coefficient dict
    parameter in: coeffs_fy_camber    ... Fy camber coefficient dict
    returns: slip_array, dict mapping fz_newton -> (fx_array, fy_array)
    """
    slip_array = np.linspace(SLIP_RANGE_START, SLIP_RANGE_END, SLIP_STEPS)
    curves = {}
    for fz_newton in fz_levels_newton:
        fx_values = np.zeros(SLIP_STEPS)
        fy_values = np.zeros(SLIP_STEPS)
        for idx, kappa_pct in enumerate(slip_array):
            fx_val, fy_val = calc_combined_forces(
                kappa_pct, slip_angle_deg, fz_newton,
                coeffs_fy, coeffs_fx, coeffs_fy_camber
            )
            fx_values[idx] = fx_val
            fy_values[idx] = fy_val
        curves[fz_newton] = (fx_values, fy_values)
    return slip_array, curves


# ==========================================================
# Plotting – replicates style of Figs. 19 and 20 in [1]
# ==========================================================
def plot_force_curves(slip_array, curves, slip_angle_deg, vehicle_weight_kg):
    """Create and save the required plot matching Figs. 19/20 of [1].

    param in: slip_array         ... 1-D kappa array (0-100 %)
    param in: curves             ... dict fz_newton -> (fx_array, fy_array)
    param in: slip_angle_deg     ... the slip angle used
    param in: vehicle_weight_kg  ... vehicle mass in kg
    """
    fig = plt.figure(figsize=(9, 7), facecolor="white")
    axis = fig.add_subplot(1, 1, 1)
    axis.set_facecolor("white")

    line_styles = ["-", "-", "-", "-"]
    line_widths  = [2.2, 1.9, 1.6, 1.3]
    gray_shades  = ["#111111", "#333333", "#666666", "#999999"]

    sorted_fz = sorted(curves.keys(), reverse=True)

    for curve_idx, fz_newton in enumerate(sorted_fz):
        fx_arr, fy_arr = curves[fz_newton]
        color = gray_shades[curve_idx]
        lw = line_widths[curve_idx]
        label_num = curve_idx + 1

        # Fx: solid line
        axis.plot(slip_array, fx_arr,
                  linestyle=line_styles[curve_idx],
                  color=color, linewidth=lw)
        # Fy: dashed line
        axis.plot(slip_array, fy_arr,
                  linestyle="--",
                  color=color, linewidth=lw * 0.85)

        # Curve number label near end of Fx (right side of plot)
        last_fx = fx_arr[-1]
        axis.annotate(
            str(label_num),
            xy=(slip_array[-1] - 2, last_fx),
            fontsize=9, color=color, va="center", ha="right"
        )

    # Axis labels and region text
    axis.set_xlabel("Longitudinal slip – κ [%]", fontsize=11)
    axis.set_ylabel("Side force Fy [N]  |  Brake force –Fx [N]", fontsize=11)
    axis.set_xlim(SLIP_RANGE_START, SLIP_RANGE_END)
    axis.set_ylim(0, 8500)
    axis.set_yticks([0, 2000, 4000, 6000, 8000])
    axis.grid(True, color="#e0e0e0", linewidth=0.5)

    # Fx/Fy region annotations (matching the paper)
    axis.text(60, 2800, "–Fx", fontsize=11, color="#333333", style="italic")
    axis.text(75, 350,  "Fy", fontsize=11, color="#333333", style="italic")

    # Legend (top right, matching paper style)
    legend_handles = [
        plt.Line2D([0], [0], color="#111111", lw=2.0, linestyle="-",
                   label="Calculations (Fx)"),
        plt.Line2D([0], [0], color="#111111", lw=1.6, linestyle="--",
                   label="Calculations (Fy)"),
    ]
    for curve_idx, fz_newton in enumerate(sorted_fz):
        fz_label = f"{curve_idx+1} – Fz = {int(round(fz_newton))} N"
        legend_handles.append(
            plt.Line2D([0], [0], color=gray_shades[curve_idx],
                       lw=line_widths[curve_idx], label=fz_label)
        )

    axis.legend(
        handles=legend_handles,
        loc="upper right", fontsize=8.5,
        frameon=True, framealpha=0.95,
        title=f"α = {slip_angle_deg} deg",
        title_fontsize=9
    )

    # Title
    fz_wheel = calc_wheel_load(vehicle_weight_kg)
    title_str = (
        f"Side force Fy and Brake force –Fx  vs  Longitudinal slip  "
        f"at {slip_angle_deg}° slip angle\n"
        f"Vehicle mass = {vehicle_weight_kg} kg  "
        f"(Fz/wheel ≈ {fz_wheel/1000:.2f} kN)  –  "
        f"Pacejka Magic Formula [1]"
    )
    axis.set_title(title_str, fontsize=9.5, pad=10)

    fig.tight_layout(pad=2.0)

    output_filename = (
        f"magic_formula_alpha{slip_angle_deg}_w{vehicle_weight_kg}.png"
    )
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), output_filename
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"[INFO] Plot saved to: {output_path}")
    return output_path


# ==========================================================
# Argument parsing
# ==========================================================
def parse_arguments():
    """Parse command-line arguments for the tyre simulation.

    returns: parsed Namespace with slip, weight, mu
    """
    arg_parser = argparse.ArgumentParser(
        description=(
            "Pacejka Magic Formula tyre model (combined slip). "
            "Plots Fy and Fx vs longitudinal slip with Fz as parameter. "
            "Ref: [1] Bakker, Nyborg, Pacejka – SAE Transactions 1987."
        )
    )
    arg_parser.add_argument(
        "--slip",
        type=float,
        default=2.0,
        help="Slip angle alpha in degrees (e.g. 2 or 5). Default: 2.0"
    )
    arg_parser.add_argument(
        "--weight",
        type=float,
        default=1500.0,
        help="Total vehicle mass in kg. Default: 1500.0"
    )
    arg_parser.add_argument(
        "--mu",
        type=float,
        default=0.8,
        help="Road friction coefficient mu (0-1). Default: 0.8"
    )
    return arg_parser.parse_args()


# ==========================================================
# Main entry point
# ==========================================================
def main():
    """Main routine: parse args, load coefficients, compute forces, plot."""
    cmd_args = parse_arguments()

    slip_angle_deg = cmd_args.slip
    vehicle_weight_kg = cmd_args.weight

    print("[INFO] Loading tyre coefficients from JSON ...")
    coeffs_fy, coeffs_fx, coeffs_fy_camber = load_coefficients(COEFFICIENTS_FILE)

    fz_wheel = calc_wheel_load(vehicle_weight_kg)
    print(f"[INFO] Vehicle mass   : {vehicle_weight_kg} kg")
    print(f"[INFO] Fz per wheel   : {fz_wheel:.1f} N  ({fz_wheel/1000:.3f} kN)")
    print(f"[INFO] Slip angle (α) : {slip_angle_deg}°")
    print(f"[INFO] Fz levels (N)  : {FZ_LEVELS_NEWTON}")

    print("[INFO] Computing combined-slip force curves ...")
    slip_array, curves = build_force_curves(
        slip_angle_deg, FZ_LEVELS_NEWTON,
        coeffs_fy, coeffs_fx, coeffs_fy_camber
    )

    print("[INFO] Generating plot ...")
    saved_path = plot_force_curves(
        slip_array, curves, slip_angle_deg, vehicle_weight_kg
    )
    print(f"[INFO] Done. Saved to: {saved_path}")


if __name__ == "__main__":
    main()
