class PotentialType:
    """A compsite variable for interactions."""

    def __init__(self, pot, vir, ovr):
        self.pot = pot # Potential Energy
        self.vir = vir # Virial
        self.ovr = ovr # Check if overlap

    def __add__(self, other):
        pot = self.pot + other.pot
        vir = self.vir + other.vir
        ovr = self.ovr or other.ovr
        return PotentialType(pot, vir, ovr)

    def __sub__(self, other):
        pot = self.pot - other.pot
        vir = self.vir - other.vir
        ovr = self.ovr or other.ovr
        return PotentialType(pot, vir, ovr)
