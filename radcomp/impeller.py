import math
from dataclasses import InitVar, dataclass, field
from math import atan, cos, pi, sin, tan
from typing import List

from scipy import optimize

from .condition import OperatingCondition
from .correlations import moody
from .geometry import Geometry
from .inducer import Inducer, InducerState
from .thermo import ThermoException, ThermoProp, static_from_total, total_from_static


@dataclass
class ImpellerState(InducerState):
    relative: ThermoProp = field(default_factory=ThermoProp)
    w: float = math.nan
    ws: float = math.nan
    m_abs_m: float = math.nan
    m_rel: float = math.nan
    m_rels: float = math.nan
    beta: float = math.nan


@dataclass
class ImpellerLosses:
    skin_friction: float = math.nan
    blade_loading: float = math.nan
    clearance: float = math.nan
    incidence: float = math.nan
    disc_friction: float = math.nan
    recirculation: float = math.nan


@dataclass
class Impeller:
    geom: InitVar[Geometry]
    op: InitVar[OperatingCondition]
    ind: InitVar[Inducer]
    in2: ImpellerState = field(init=False)
    in3: ImpellerState = field(default_factory=ImpellerState)
    out: ImpellerState = field(default_factory=ImpellerState)
    losses: ImpellerLosses = field(default_factory=ImpellerLosses)
    dh0s: float = math.nan
    eff: float = math.nan
    choke_flag = False
    wet = False

    def __post_init__(
        self, geom: Geometry, op: OperatingCondition, ind: Inducer
    ) -> None:
        self.in2 = ImpellerState.from_state(ind.out)
        if self.out.is_not_set:
            self.calculate(geom, op)

    def skin_friction_losses(self, geom: Geometry, w4: float, tp4: ThermoProp) -> float:
        """Calculates the impeller skin friction losses according to Jansen, Coppage Galvas"""
        Dh, Lh = geom.hydraulic_diameter
        Re = (
            Dh
            * (self.in2.w + w4)
            / 2
            * (self.in2.static.D + tp4.D)
            / 2
            / ((self.in2.static.V + tp4.V) / 2)
        )
        Cf = moody(Re, geom.rug_imp / Dh)
        return 4 * Cf * Lh * ((self.in2.w + w4) / 2) ** 2 / (2 * Dh)

    def diffusion_factor(
        self, geom: Geometry, out_H: float, w4: float, n_rot: float
    ) -> float:
        _, Lh = geom.hydraulic_diameter
        wx = self.in2.w
        dh_aero = (out_H) / (n_rot * geom.r4) ** 2
        # Coppage and Rodgers
        Df = (
            1
            - w4 / wx
            + pi
            * geom.r4**2
            * dh_aero
            * n_rot
            / ((geom.n_blades + geom.n_splits) * Lh * wx)
            + 0.1
            * (geom.r2s - geom.r2h + geom.b4)
            / 2
            / (geom.r4 - geom.r2s)
            * (1 + w4 / wx)
        )
        return Df

    def blade_loading_losses(self, geom: Geometry, Df: float, n_rot: float) -> float:
        """Calculates the impeller blade loading and diffusion losses according to Rodgers"""
        return 0.05 * Df**2 * (n_rot * geom.r4) ** 2

    def clearance_losses(
        self, geom: Geometry, tp4: ThermoProp, c4t: float, n_rot: float
    ) -> float:
        """Calculates the impeller tip clearance losses according to Jansen, Brasz"""
        c4t = abs(c4t)
        tip_speed = n_rot * geom.r4
        return (
            0.6
            * geom.clearance
            / geom.b4
            * c4t
            / tip_speed
            * (
                4
                * pi
                / geom.b4
                / geom.n_blades
                * c4t
                * self.in2.c
                * cos(geom.alpha2 / 180 * pi)
                / tip_speed**2
                * (geom.r2s**2 - geom.r2h**2)
                / ((geom.r4 - geom.r2s) * (1 + tp4.D / self.in2.static.D))
            )
            ** 0.5
            * tip_speed**2
        )

    def disc_friction_losses(
        self, geom: Geometry, tp4: ThermoProp, m: float, n_rot: float
    ) -> float:
        """Calculates the impeller disc friction losses according to Daily & Nece"""
        if tp4.V < 0:
            print("Negative V", tp4)
        Re_y = 2.0 * n_rot * geom.r4**2 * tp4.D / tp4.V
        if Re_y > 3e5:
            Kf = 0.102 * (geom.backface / geom.r4) ** 0.1 / Re_y**0.2
        else:
            Kf = 3.7 * (geom.backface / geom.r4) ** 0.1 / Re_y**0.5
        return 0.25 * (tp4.D) * n_rot * geom.r4**3 * Kf / m * (n_rot * geom.r4) ** 2

    def recirculation_losses(
        self, geom: Geometry, Df: float, alpha: float, n_rot: float
    ) -> float:
        """Calculates the impeller recirculation losses according to Coppage"""
        return 0.02 * Df**2 * tan(abs(alpha / 180 * pi)) * (n_rot * geom.r4) ** 2

    def calculate(self, geom: Geometry, op: OperatingCondition) -> None:
        # Calculate the relative point at the impeller inlet (Point2)
        c2_theta = self.in2.c * sin(geom.alpha2 * pi / 180)
        c2_m = self.in2.c * cos(geom.alpha2 * pi / 180)
        w2t_s = geom.r2s * op.n_rot - c2_theta  # @ shroud
        beta2_fs = -atan(w2t_s / c2_m) * 180 / pi  # @ shroud
        w2_s = c2_m / cos(beta2_fs / 180 * pi)  # % @ shroud
        # Relative speed and flow angle at rms radius
        w2t = geom.r2rms * op.n_rot - c2_theta
        beta2_f = -atan(w2t / c2_m) * 180 / pi
        w2 = c2_m / cos(beta2_f / 180 * pi)
        # Thermodynamic relative impeller inlet point
        try:
            self.in2.relative = total_from_static(self.in2.static, w2)
        except ThermoException:
            self.wet = True
            return

        # self.in2.beta_fs = beta2_fs
        # self.in2.beta_f  = beta2_f
        self.in2.ws = w2_s
        self.in2.w = w2
        self.in2.m_rel = w2 / self.in2.static.A
        self.in2.m_rels = w2_s / self.in2.static.A

        if self.in2.m_rel >= 0.99:
            self.choke_flag = True
            return

        beta2_opt = atan(geom.A_x / geom.A_y * tan(geom.beta2 / 180 * pi)) * 180 / pi
        # according to Stanitz-Galvas
        dh_inc = 0.5 * (w2 * sin(abs(abs(beta2_f) - abs(beta2_opt)) / 180 * pi)) ** 2
        try:
            rel3_temp = op.fld.thermo_prop(
                "HS", self.in2.relative.H - dh_inc, self.in2.relative.S
            )
        except ThermoException:
            self.choke_flag = True
            return
        self.in3.relative = op.fld.thermo_prop("PH", rel3_temp.P, self.in2.relative.H)

        self.losses.incidence = dh_inc

        # Resolve static 3
        def resolve_static(x):
            try:
                stat3 = static_from_total(self.in2.relative, x)
            except ThermoException:
                return 1e4

            return (op.m - geom.A_y * x * stat3.D) / op.m

        w3_guess = 0.65 * self.in2.relative.A
        # w_guess = ind.m / geom.A_y / self.in2.relative.D
        sol = optimize.root(resolve_static, x0=w3_guess)
        if (sol.fun > 0.001).any():
            self.choke_flag = True
            return

        w3_throat = sol.x[0]
        self.in3.static = static_from_total(self.in2.relative, w3_throat)

        c3_m = c2_m * geom.A_x / geom.A_y
        c3 = c3_m / cos(geom.alpha2 * pi / 180)
        self.in3.m_rel = w3_throat / self.in3.static.A
        self.in3.m_abs = c3 / self.in3.static.A

        # Inlet checks
        self.in3.total = total_from_static(self.in3.static, c3)
        self.in3.w = w3_throat
        self.in3.c = c3

        # Impeller Discharge
        h4_rel = (
            0.5 * ((geom.r4 * op.n_rot) ** 2 - (geom.r2rms * op.n_rot) ** 2)
            + self.in2.relative.H
        )
        try:
            tp4_rel = op.fld.thermo_prop("HS", h4_rel, self.in2.relative.S)
        except ThermoException:
            self.choke_flag = True
            return

        if tp4_rel.phase == "twophase":
            # Check if impeller operates in wet area.
            # If yes, rise flags and abandon ship!
            self.wet = True
            return
        A4_total = 2 * pi * geom.r4 * geom.b4 * geom.blockage[3]

        def resolve_discharge_triangle(x: List[float]) -> List[float]:
            beta4_f, w4, dh_losses, p4_rel = x

            dh_lo = dh_losses
            if dh_losses < 0:
                dh_lo = 0

            p4r = p4_rel
            if p4_rel <= 0:
                p4r = tp4_rel.P

            err = []
            try:
                tp4_r = op.fld.thermo_prop("PH", p4r, h4_rel + dh_lo)
                # Part 1 Triangle Discharge
                A4_rel = A4_total * cos(beta4_f * pi / 180)

                tp4_stat = static_from_total(tp4_r, w4)

                err.append((op.m - A4_rel * w4 * tp4_stat.D) / op.m)

                c4m = op.m / A4_total / tp4_stat.D
                c4t = c4m * tan(geom.beta4 / 180 * pi) + geom.slip * (
                    geom.r4 * op.n_rot
                )
                w4t = (geom.r4 * op.n_rot) - c4t

                w4_new = (w4t**2 + c4m**2) ** 0.5
                beta4_f_new = -math.asin(w4t / w4_new) * 180 / pi
                err.append((beta4_f_new - beta4_f) / 60.0)

                # Part 2 Pressure
                c4 = (c4t**2 + c4m**2) ** 0.5
                alpha = atan(c4t / c4m) * 180 / pi

                tp4_tot = total_from_static(tp4_stat, c4)
                out_H = tp4_tot.H - self.in2.total.H
                Df = self.diffusion_factor(geom, out_H, w4, op.n_rot)

                # Calculate internal losses
                dh_sf = self.skin_friction_losses(geom, w4, tp4_stat)
                dh_bl = self.blade_loading_losses(geom, Df, op.n_rot)
                dh_cl = self.clearance_losses(geom, tp4_stat, c4t, op.n_rot)
                dh_losses_int = dh_sf + dh_bl + dh_cl + dh_inc

                # Calculate external losses
                dh_df = self.disc_friction_losses(geom, tp4_stat, op.m, op.n_rot)
                dh_r = self.recirculation_losses(geom, Df, alpha, op.n_rot)
                dh_losses_ext = dh_df + dh_r

                err.append((dh_losses_ext - dh_losses) / self.in2.relative.H)

                # Correct pressure
                tp4_temp = op.fld.thermo_prop(
                    "HS", h4_rel - dh_losses_int, self.in2.relative.S
                )

                err.append(
                    (tp4_temp.P - tp4_r.P) / self.in2.relative.P + (abs(p4_rel - p4r))
                )

            except ThermoException:
                err.extend([1e4] * (4 - len(err)))

            return err

        # Guesses
        beta4_f0 = geom.beta4 - 10.0
        A4_rel = (
            2 * pi * geom.r4 * geom.b4 * geom.blockage[3] * cos(beta4_f0 * pi / 180)
        )
        w4_guess = op.m / A4_rel / tp4_rel.D

        dh_df_guess = self.disc_friction_losses(geom, tp4_rel, op.m, op.n_rot)

        sol = optimize.root(
            resolve_discharge_triangle,
            x0=[beta4_f0, w4_guess, dh_df_guess, tp4_rel.P],
            tol=1e-4,
        )
        if (sol.fun > 0.001).any():
            self.choke_flag = True
            return

        beta4_f, w4, dh_losses, p4_rel = sol.x
        self.out.w = w4
        self.out.relative = op.fld.thermo_prop("PH", p4_rel, h4_rel + dh_losses)
        self.out.static = static_from_total(self.out.relative, w4)

        c4m = op.m / A4_total / self.out.static.D
        c4t = c4m * tan(geom.beta4 / 180 * pi) + geom.slip * (geom.r4 * op.n_rot)
        # w4t = (geom.r4*op.n_rot) - c4t
        c4 = (c4t**2 + c4m**2) ** 0.5
        self.out.c = c4
        alpha = atan(c4t / c4m) * 180 / pi
        try:
            self.out.total = total_from_static(self.out.static, c4)
        except:
            print(c4, w4)
            raise
        self.out.isentropic = op.fld.thermo_prop(
            "PS", self.out.total.P, self.in2.static.S
        )

        out_H = self.out.total.H - self.in2.total.H
        Df = self.diffusion_factor(geom, out_H, w4, op.n_rot)

        self.losses.skin_friction = self.skin_friction_losses(geom, w4, self.out.static)
        self.losses.blade_loading = self.blade_loading_losses(geom, Df, op.n_rot)
        self.losses.clearance = self.clearance_losses(
            geom, self.out.static, c4t, op.n_rot
        )

        self.losses.disc_friction = self.disc_friction_losses(
            geom, self.out.static, op.m, op.n_rot
        )
        self.losses.recirculation = self.recirculation_losses(geom, Df, alpha, op.n_rot)

        self.out.m_abs = c4 / self.out.static.A
        self.out.m_abs_m = c4 * cos(alpha * pi / 180) / self.out.static.A
        self.out.m_rel = w4 / self.out.static.A

        self.out.beta = beta4_f
        self.out.alpha = alpha

        self.dh0s = self.out.isentropic.H - self.in2.total.H
        self.eff = out_H / self.dh0s

        if self.out.m_rel >= 0.99 or self.out.m_abs_m >= 0.99:
            self.choke_flag = True

        if self.out.total.P < self.in2.total.P:
            self.choke_flag = True
