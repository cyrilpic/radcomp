"""Set of functions and classes to use CoolProp as backend"""
__all__ = ["CoolPropFluid"]

from dataclasses import dataclass
from typing import Union

import CoolProp as CP

from .base import Fluid, ThermoException, ThermoProp


cp_inputs = {
    "PT": (CP.iP, CP.iT),
    "HS": (CP.iHmass, CP.iSmass),
    "PH": (CP.iP, CP.iHmass),
    "PS": (CP.iP, CP.iSmass),
    "TQ": (CP.iT, CP.iQ),
    "PQ": (CP.iP, CP.iQ),
}

cp_outputs = {"P": CP.iP, "T": CP.iT, "D": CP.iDmass, "H": CP.iHmass, "S": CP.iSmass}


cp_phases = {
    CP.iphase_gas: "gas",
    CP.iphase_twophase: "twophase",
    CP.iphase_supercritical: "supercritical",
    CP.iphase_supercritical_gas: "supercritical_gas",
}


@dataclass
class CoolPropFluid(Fluid):
    name: str

    def __post_init__(self):
        self.state = CP.AbstractState("HEOS", self.name.upper())

    @property
    def P_max(self) -> float:
        return self.state.pmax()

    @property
    def T_max(self) -> float:
        return self.state.Tmax()

    @property
    def P_crit(self) -> float:
        return self.state.p_critical()

    @property
    def T_crit(self) -> float:
        return self.state.T_critical()

    @property
    def P_triple(self) -> float:
        return self.state.keyed_output(CP.iP_triple)

    @property
    def T_triple(self) -> float:
        return self.state.Ttriple()

    def thermo_prop(
        self, in_type: Union[str, int], in1: float, in2: float
    ) -> "ThermoProp":
        if isinstance(in_type, str):
            inputs = cp_inputs[in_type]
            input_pair = CP.CoolProp.generate_update_pair(
                inputs[0], in1, inputs[1], in2
            )
        else:
            input_pair = (in_type, in1, in2)

        try:
            self.state.update(*input_pair)
        except ValueError as e:
            raise ThermoException(*e.args, *input_pair)

        if self.state.phase() not in cp_phases.keys():
            raise ThermoException("Not gas or two-phase")

        d = {k: self.state.keyed_output(v) for k, v in cp_outputs.items()}
        d["phase"] = cp_phases[self.state.phase()]
        if self.state.phase() == CP.iphase_twophase:
            d["A"] = self.state.saturated_vapor_keyed_output(CP.ispeed_sound)
            d["V"] = self.state.saturated_vapor_keyed_output(CP.iviscosity)
        else:
            d["A"] = self.state.keyed_output(CP.ispeed_sound)
            d["V"] = self.state.keyed_output(CP.iviscosity)
        d["fld"] = self

        return ThermoProp(**d)

    def __getstate__(self):
        return {"name": self.name}
