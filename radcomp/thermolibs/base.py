__all__ = ["Fluid", "ThermoException", "ThermoProp"]

from dataclasses import dataclass, field
from math import nan


class ThermoException(Exception):
    "Thermodynamic Error"


class Fluid:
    """Abstract base class for fluids"""

    def activate(self):
        pass

    def thermo_prop(self, in_type: str, in1: float, in2: float) -> "ThermoProp":
        return ThermoProp(fld=self)


@dataclass(frozen=True)
class ThermoProp:
    """Thermodynamic properties storage class"""

    P: float = nan
    T: float = nan
    D: float = nan
    H: float = nan
    S: float = nan
    A: float = nan
    V: float = nan
    phase: str = ""
    fld: Fluid = field(default_factory=Fluid)
