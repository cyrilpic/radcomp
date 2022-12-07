import time
from typing import List, Tuple, Iterator

import numpy as np

from .compressor import Compressor
from .condition import OperatingCondition
from .geometry import Geometry
from .thermo import ThermoProp


def scaled_geometry(r4, beta4, b4r4, n_blades, r2sor4, r2sor2h, Cmin, CR):
    return Geometry(
        r1=1.1*r2sor4*r4,
        beta2=-45,
        beta2s=-60,
        alpha2=0.,
        r4=r4,
        r2s=r2sor4*r4,
        r2h=r2sor4*r4/r2sor2h,
        b4=b4r4*r4,
        beta4=beta4,
        n_blades=n_blades,
        n_splits=n_blades,
        r5=1.5*r4,
        b5=b4r4*r4,
        blade_e=0.2e-3,
        rug_imp=1.2e-5,
        clearance=max(Cmin, CR*b4r4*r4),
        backface=max(Cmin, CR*b4r4*r4),
        ind_rug=1.2e-5,
        l_ind=4*r4,
        l_comp=0.7*r4,
        blockage=[1., 1., 1., 1., 1.]
    )


def upper_bounds(geom: Geometry, in0: ThermoProp, max_tipspeed=600, max_mach=1):
    n_rot_max = max_tipspeed/geom.r4

    mflow_max = max_mach * in0.A * in0.D * geom.A2_eff

    return n_rot_max, mflow_max


def calculate_on_op_grid(geom: Geometry, in0: ThermoProp, lb: np.ndarray,
                         ub: np.ndarray, resolution=0.005,
                         map_func=map) -> Tuple[np.ndarray, Iterator]:
    xx, yy = np.mgrid[0:1:resolution, 0:1:resolution]
    grid = np.c_[xx.ravel(), yy.ravel()]

    X = grid * (ub - lb) + lb

    def calculate_compressor(x: List[float]) -> Tuple[Compressor, float]:
        n_rot, m = x
        op = OperatingCondition(in0=in0, fld=in0.fld,
                                m=m, n_rot=n_rot/30*np.pi)
        t0 = time.perf_counter()
        comp = Compressor(geom, op)
        comp.calculate()
        dt = time.perf_counter() - t0
        return comp, dt

    return X, map_func(calculate_compressor, X.tolist())
