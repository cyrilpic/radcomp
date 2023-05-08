"""Set of functions and classes to use RefProp as backend

The assumption is that RefProp is located at env['RPPREFIX'] or
env['RPLIBRARY'].

Fluids need to be activate before they can be used.
"""
__all__ = ["RefpropFluid"]

import os
from dataclasses import dataclass, field

from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary

from .base import Fluid, ThermoException, ThermoProp


RP = REFPROPFunctionLibrary(os.environ.get("RPLIBRARY", os.environ["RPPREFIX"]))
RP.SETPATHdll(os.environ["RPPREFIX"])
print(f"Using RefProp: {RP.RPVersion()}")
MASS_BASE_SI = RP.GETENUMdll(0, "MASS BASE SI").iEnum


refprop_phase = {999: "supercritical", 998: "supercritical_gas"}


@dataclass
class RefpropFluid(Fluid):
    name: str
    P_max: float = field(init=False)
    T_max: float = field(init=False)
    P_crit: float = field(init=False)
    T_crit: float = field(init=False)
    P_triple: float = field(init=False)
    T_triple: float = field(init=False)

    def __post_init__(self):
        r = RP.REFPROPdll(
            self.name,
            "",
            "PMAX;TMAX;Pc;Tc;PTRP;TTRP",
            MASS_BASE_SI,
            0,
            0,
            0.0,
            0.0,
            [1.0],
        )
        if r.ierr != 0:
            raise ThermoException(r.herr)
        self.P_max = r.Output[0]
        self.T_max = r.Output[1]
        self.P_crit = r.Output[2]
        self.T_crit = r.Output[3]
        self.P_triple = r.Output[4]
        self.T_triple = r.Output[5]

    def activate(self):
        RP.SETFLUIDSdll(self.name)

    def thermo_prop(self, in_type, in1, in2):
        r = RP.REFPROPdll("", in_type, "P;T;D;H;S", MASS_BASE_SI, 0, 0, in1, in2, [1.0])

        if r.ierr != 0:
            raise ThermoException(r.herr)

        d = dict(zip(["P", "T", "D", "H", "S"], r.Output))

        if r.q >= 0 and r.q < 1:
            d["phase"] = "twophase"
            r = RP.REFPROPdll("", "PQ", "W;VIS", MASS_BASE_SI, 0, 0, d["P"], 1.0, [1.0])
        elif r.q < 0:
            raise ThermoException("Liquid")
        else:
            if r.q == 998:
                d["phase"] = "supercritical_gas"
            elif r.q == 999:
                d["phase"] = "supercritical"
            else:
                d["phase"] = "gas"
            r = RP.REFPROPdll(
                "", "PT", "W;VIS", MASS_BASE_SI, 0, 0, d["P"], d["T"], [1.0]
            )

        d["A"] = r.Output[0]
        d["V"] = r.Output[1]
        d["fld"] = self

        return ThermoProp(**d)
