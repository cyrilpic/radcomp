__all__ = [
    "CoolPropFluid",
    "Fluid",
    "RefpropFluid",
    "ThermoException",
    "ThermoProp",
    "static_from_total",
    "total_from_static",
]

from .thermolibs.base import Fluid, ThermoException, ThermoProp

try:
    from .thermolibs.coolprop import CoolPropFluid
except ImportError as e:
    CoolPropFluid = None

try:
    from .thermolibs.refprop import RefpropFluid
except (ImportError, KeyError) as e:
    RefpropFluid = None


if CoolPropFluid is None and RefpropFluid is None:
    raise ImportError(
        "No thermodynamic library available. Install CoolProp or Refprop."
    )


def static_from_total(tot: ThermoProp, speed: float) -> ThermoProp:
    """Get static flow condition based on total condition and flow speed"""
    return tot.fld.thermo_prop("HS", (tot.H - 0.5 * speed**2), tot.S)


def total_from_static(stat: ThermoProp, speed: float) -> ThermoProp:
    """Get total flow condition based on static condition and flow speed"""
    return stat.fld.thermo_prop("HS", (stat.H + 0.5 * speed**2), stat.S)
