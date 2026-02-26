import json
import numpy as np


class TyreModel:
    """
    Implements the Pacejka Magic Formula tyre model.
    """

    def __init__(self, coeffs_path: str, mu: float):
        self.mu = mu
        self.coeffs = self._load_coeffs(coeffs_path)

    def _load_coeffs(self, path: str) -> dict:
        """
        Load tyre coefficients from a JSON file.
        """
        with open(path, "r") as f:
            return json.load(f)

    def _magic_formula(self, x: np.ndarray, B: float, C: float, D: float, E: float) -> np.ndarray:
        """
        Standard Pacejka Magic Formula.
        """
        return D * np.sin(
            C * np.arctan(B * x - E * (B * x - np.arctan(B * x)))
        )

    def longitudinal_force(self, slip: np.ndarray, Fz: float) -> np.ndarray:
        """
        Compute longitudinal (brake) force Fx as a function of longitudinal slip.
        """
        params = self.coeffs["longitudinal"]

        C = params["C"]
        E = params["E"]
        D = self.mu * Fz * params["D_factor"]
        B = params["B"] / (C * D)

        return self._magic_formula(slip, B, C, D, E)

    def lateral_force(self, slip_angle_rad: float, slip: np.ndarray, Fz: float) -> np.ndarray:
        """
        Compute lateral force Fy with longitudinal slip as parameter.
        """
        params = self.coeffs["lateral"]

        C = params["C"]
        E = params["E"]
        D0 = self.mu * Fz * params["D_factor"]
        B = params["B"] / (C * D0)

        Fy = np.zeros_like(slip)

        for i, kappa in enumerate(slip):
            # Simple combined-slip reduction (acceptable for assignment)
            reduction = max(0.0, 1.0 - params["combined_reduction"] * kappa)
            D = D0 * reduction

            Fy[i] = self._magic_formula(
                np.array([slip_angle_rad]), B, C, D, E
            )[0]

        return Fy
