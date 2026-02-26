import json
import numpy as np


class TyreModel:
    """
    Pacejka tyre model using Bakker coefficients (Tables 2 & 3).
    """

    def __init__(self, coeffs_path: str):
        with open(coeffs_path, "r") as f:
            self.c = json.load(f)

    def _magic_formula(self, B, C, D, E, phi):
        return D * np.sin(C * np.arctan(B * phi))

    # ---------------- Longitudinal Force ----------------
    def longitudinal_force(self, kappa: np.ndarray, Fz: float):
        """
        Fz in Newton → converted to kN internally
        """
        Fz = Fz / 1000.0  # kN

        p = self.c["Fx"]

        D = p["a1"] * Fz**2 + p["a2"] * Fz
        C = p["C"]
        B = (p["a3"] * Fz**2 + p["a4"] * Fz) / (C * D * np.exp(p["a5"] * Fz))
        E = p["a6"] * Fz**2 + p["a7"] * Fz + p["a8"]

        phi = (1 - E) * kappa + (E / B) * np.arctan(B * kappa)

        return self._magic_formula(B, C, D, E, phi)

    # ---------------- Lateral Force ----------------
    def lateral_force(self, alpha: float, kappa: np.ndarray, Fz: float, gamma: float = 0.0):
        """
        alpha, gamma in radians
        """
        Fz = Fz / 1000.0  # kN

        p = self.c["Fy"]
        cam = self.c["camber"]

        D = p["a1"] * Fz**2 + p["a2"] * Fz
        C = p["C"]
        B = (
            p["a3"] * np.sin(p["a4"] * np.arctan(p["a5"] * Fz))
            / (C * D)
        ) * (1 - cam["a12"] * abs(gamma))

        E = p["a6"] * Fz**2 + p["a7"] * Fz + p["a8"]

        dSh = cam["a9"] * gamma
        dSv = (cam["a10"] * Fz**2 + cam["a11"] * Fz) * gamma

        Fy = np.zeros_like(kappa)

        for i, _ in enumerate(kappa):
            phi = (1 - E) * (alpha + dSh) + (E / B) * np.arctan(B * (alpha + dSh))
            Fy[i] = self._magic_formula(B, C, D, E, phi) + dSv

        return Fy
