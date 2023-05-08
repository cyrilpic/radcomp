import math

from .condition import OperatingCondition
from .diffuser import VanelessDiffuser, surge_critical_angle
from .geometry import Geometry
from .impeller import Impeller
from .inducer import Inducer


class Compressor:
    def __init__(self, geom: Geometry, op: OperatingCondition):
        self.geom = geom
        self.op = op

        self.ind = None
        self.imp = None
        self.dif = None

        self.in_ = None
        self.out = None

        self.invalid_flag = False
        self.eff = math.nan
        self.dh0s = math.nan
        self.PR = math.nan
        self.power = math.nan
        self.Ns = math.nan
        self.Ds = math.nan
        self.m_in = math.nan
        self.head = math.nan
        self.d_head_d_flow = math.nan
        self.tip_speed = geom.r4 * op.n_rot
        self.n_rot_corr = self.tip_speed / op.in0.A
        self.V_in = self.op.m / op.in0.D
        self.flow = self.V_in / (self.tip_speed * self.geom.r4**2)

    def calculate(self, delta_check=True) -> bool:
        # Inducer
        self.ind = Inducer(self.geom, self.op)
        if self.ind.choke_flag:
            self.invalid_flag = True
            return False
        self.in_ = self.ind.in1
        self.m_in = self.ind.out.c / self.in_.total.A

        # Impeller
        self.imp = Impeller(self.geom, self.op, self.ind)
        if self.imp.choke_flag or self.imp.wet:
            self.invalid_flag = True
            return False

        # Check surge
        alpha_crit = surge_critical_angle(
            self.geom.r5, self.geom.r4, self.geom.b4, self.imp.out.m_abs
        )
        if self.imp.out.alpha > alpha_crit:
            self.invalid_flag = True
            return False

        # Diffuser
        self.dif = VanelessDiffuser(self.geom, self.op, self.imp)
        if self.dif.choke_flag:
            self.invalid_flag = True
            return False

        # No volute
        self.out = self.dif.out

        # Final calculations
        dh = self.out.total.H - self.in_.total.H
        PR = self.out.total.P / self.in_.total.P
        if dh < 0 or PR < 1:
            self.invalid_flag = True
            return False

        tp_is = self.op.fld.thermo_prop("PS", self.out.total.P, self.in_.total.S)
        self.dh0s = tp_is.H - self.in_.total.H
        self.head = self.dh0s / (self.tip_speed**2)

        # Assess surge by calculating dHead/dFlow should be < 0
        if delta_check:
            d_op = OperatingCondition(**self.op.__dict__)
            d_op.m *= 1.005
            d_comp = Compressor(self.geom, d_op)
            if d_comp.calculate(delta_check=False):
                self.d_head_d_flow = (d_comp.head - self.head) / (
                    d_comp.flow - self.flow
                )
                if self.d_head_d_flow > -1e-4:
                    self.invalid_flag = True
                    return False

        self.eff = self.dh0s / dh
        self.PR = PR
        self.power = self.op.m * dh
        sqrt_v_in = self.V_in**0.5
        self.Ns = self.op.n_rot * sqrt_v_in / (self.dh0s**0.75)
        self.Ds = 2 * self.geom.r4 * self.dh0s**0.25 / sqrt_v_in

        return not self.invalid_flag
