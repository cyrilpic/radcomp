import math
from dataclasses import InitVar, dataclass, field
from math import cos, pi, sin, tan

import numpy as np
from numpy.polynomial import polynomial
from scipy import optimize

from .condition import OperatingCondition
from .geometry import Geometry
from .impeller import Impeller
from .inducer import InducerState
from .thermo import ThermoException, static_from_total


class VanelessState(InducerState):
    pass


@dataclass
class VanelessDiffuser:
    geom: InitVar[Geometry]
    op: InitVar[OperatingCondition]
    imp: InitVar[Impeller]
    in4: VanelessState = field(init=False)
    out: VanelessState = field(default_factory=VanelessState)
    loss: float = math.nan
    dh0s: float = math.nan
    eff: float = math.nan
    choke_flag = False
    n_steps: int = 15

    def __post_init__(
        self, geom: Geometry, op: OperatingCondition, imp: Impeller
    ) -> None:
        self.in4 = VanelessState.from_state(imp.out)
        self.calculate(geom, op)

    def calculate(self, geom: Geometry, op: OperatingCondition):
        r = np.linspace(geom.r4, geom.r5, 1 + self.n_steps, endpoint=True)
        dr = np.diff(r)
        b = np.linspace(geom.b4, geom.b5, 1 + self.n_steps, endpoint=True)

        Dh = np.sqrt(8 * r[:-1] * b[1:] * geom.blockage[4])  # Hydraulic
        A_eff = 2 * r[1:] * b[1:] * pi * geom.blockage[4]

        k = 0.02

        def resolve_speed(x, return_values=False):
            in_ = VanelessState.from_state(self.in4)
            err = []
            for i in range(self.n_steps):
                # Calculate friction losses
                Re = in_.c * in_.static.D / in_.static.V * b[i + 1]
                Cf = k * (1.8e5 / Re) ** 0.2  # Japikse

                # Calculate total pressure losses
                ds = (
                    (dr[i] / tan((90 - in_.alpha) / 180 * pi)) ** 2 + dr[i] ** 2
                ) ** 0.5
                dp0 = 4.0 * Cf * ds * in_.c**2 * in_.static.D / 2 / Dh[i]

                c4t = in_.c * sin(in_.alpha / 180 * pi)
                c4m = in_.c * cos(in_.alpha / 180 * pi)
                dCtdr = (
                    -(
                        c4t / r[i]
                        + Cf * in_.c**2 * sin(in_.alpha / 180 * pi) / c4m / b[i + 1]
                    )
                    * dr[i]
                )
                c5t = c4t + dCtdr

                P0 = in_.total.P - dp0
                if P0 <= 0 and P0 < op.in0.P:
                    err.extend((self.n_steps - i) * [1e4])
                    return err
                tot = op.fld.thermo_prop("PH", P0, in_.total.H)

                c5m = x[i]
                c5 = (c5m**2 + c5t**2) ** 0.5
                if c5 > 1.25 * in_.total.A:
                    # Choke
                    err.extend((self.n_steps - i) * [1e4])
                    return err

                try:
                    stat = static_from_total(tot, c5)
                except ThermoException:
                    err.extend((self.n_steps - i) * [1e4])
                    return err

                err.append((op.m - A_eff[i] * c5m * stat.D) / op.m)

                in_.c = c5
                in_.alpha = math.asin(c5t / c5) * 180 / pi
                in_.total = tot
                in_.static = stat
                in_.m_abs = in_.c * cos(in_.alpha / 180 * pi) / in_.static.A
                if in_.m_abs >= 0.99:
                    err[-1] += in_.m_abs - 0.99
            if return_values:
                return err, in_
            return err

        c4m = self.in4.c * cos(self.in4.alpha / 180 * pi)

        if c4m / self.in4.static.A >= 0.99:
            self.choke_flag = True
            return

        speed_guess = c4m * r[:-1] / r[1:]
        sol = optimize.root(resolve_speed, x0=speed_guess)

        if (sol.fun > 0.001).any():
            self.choke_flag = True
            return

        _, out = resolve_speed(sol.x, return_values=True)
        out.m_abs = out.c * cos(out.alpha / 180 * pi) / out.static.A
        if out.m_abs >= 0.99:
            self.choke_flag = True
        self.out = out

        out_is = op.fld.thermo_prop("PS", out.total.P, self.in4.total.S)
        self.out.isentropic = out_is
        self.loss = out.total.H - out_is.H
        self.dh0s = out_is.H - self.in4.total.H

        delta_h = out.total.H - self.in4.total.H
        if abs(delta_h) <= 1e-6:
            self.eff = math.copysign(math.inf, self.dh0s)
        else:
            self.eff = self.dh0s / delta_h


# Surge


def _generate_fits():
    mach_values = np.array([0, 0.4, 0.8, 1.2, 1.6])
    b_ratio = np.array([0.05, 0.1, 0.2, 0.3, 0.4])
    deg = np.array([3, 3])

    a_12 = np.array(
        [
            [80.78, 80, 78.59, 76.41, 73.9],
            [76.71, 75.47, 73.28, 70.47, 67.19],
            [73.91, 72.97, 70.63, 66.25, 60],
            [72.81, 71.87, 69.53, 64.53, 55.63],
            [72.19, 71.25, 68.75, 63.59, 54.22],
        ]
    )

    a_20 = np.array(
        [
            [80.78, 80.16, 78.59, 76.41, 73.91],
            [76.56, 77.19, 73.44, 70.63, 67.19],
            [74.06, 71.56, 68.75, 64.84, 60.31],
            [70.47, 69.38, 66.25, 61.25, 55.16],
            [69.22, 68.13, 64.84, 59.38, 52.97],
        ]
    )

    def polyfit2d(x, y, z, deg):
        xx, yy = np.meshgrid(x, y)
        lhs = polynomial.polyvander2d(xx.ravel(), yy.ravel(), deg).T
        rhs = z.ravel().T

        scl = np.sqrt(np.square(lhs).sum(1))
        scl[scl == 0] = 1

        rcond = xx.size * np.finfo(xx.dtype).eps

        c1, _, _, _ = np.linalg.lstsq(lhs.T / scl, rhs.T, rcond)
        c1 = (c1.T / scl).T

        return c1

    c12 = polyfit2d(mach_values, b_ratio, a_12, deg)
    c20 = polyfit2d(mach_values, b_ratio, a_20, deg)

    shape = deg + 1

    return c12.reshape(shape), c20.reshape(shape)


c_12, c_20 = _generate_fits()


def surge_critical_angle(r5: float, r4: float, b4: float, m2: float) -> float:
    ratio = b4 / r4
    length = r5 / r4

    angle_12 = polynomial.polyval2d(m2, ratio, c_12)
    angle_20 = polynomial.polyval2d(m2, ratio, c_20)

    alpha_r = angle_12 + (angle_20 - angle_12) * (length - 1.2) / (2.0 - 1.2)
    return 90.0 - 0.35 * (90.0 - alpha_r)
