from dataclasses import dataclass

from .thermo import Fluid, ThermoProp


@dataclass
class OperatingCondition:
    in0: ThermoProp
    fld: Fluid
    m: float
    n_rot: float  # Warning: in rad/s
