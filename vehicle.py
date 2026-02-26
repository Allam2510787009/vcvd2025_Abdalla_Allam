class Vehicle:
    """
    Represents a vehicle and computes wheel loads.
    """

    def __init__(self, mass_kg: float):
        self.mass_kg = mass_kg
        self.g = 9.81  # gravity (m/s^2)

    def total_weight(self) -> float:
        """
        Returns total vehicle weight in Newton.
        """
        return self.mass_kg * self.g

    def wheel_load(self) -> float:
        """
        Returns vertical load per wheel (equal distribution).
        """
        return self.total_weight() / 4.0
