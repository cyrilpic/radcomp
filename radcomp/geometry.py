import math
from dataclasses import dataclass, fields
from typing import List, Union


@dataclass
class Geometry:
    """Class describing the geometry of a radial compressor"""

    r1: float  # Inducer inlet radius
    r2s: float  # Shroud tip radius
    r2h: float  # Impeller hub radius
    beta2: float  # Mid-blade impeller inlet angle
    beta2s: float  # Impeller shroud angle
    alpha2: float  # Inlet flow angle
    r4: float  # Tip radius
    b4: float  # Blade height
    r5: float  # Diffuser outlet radius
    b5: float  # Diffuser passage width
    beta4: float  # Impeller outlet angle
    n_blades: int  # Number of blades
    n_splits: int  # Number of splitter blades
    blade_e: float  # Blade thickness
    rug_imp: float  # Impeller surface roughness
    clearance: float  # Tip clearance
    backface: float  # Backface clearance
    rug_ind: float  # Inducer surface roughness
    l_ind: float  # Inducer length
    l_comp: float  # Impeller length --> no impact on calculation

    blockage: List[float]

    @property
    def r2rms(self):
        return math.sqrt((self.r2s**2 + self.r2h**2) / 2.0)

    @property
    def A1_eff(self):
        return self.r1**2 * math.pi * self.blockage[0]

    @property
    def A2_eff(self):
        return (
            (self.r2s**2 - self.r2h**2)
            * math.pi
            * self.blockage[1]
            * math.cos(self.alpha2 / 180.0 * math.pi)
        )

    @property
    def A_x(self):
        # Effective area at station 2
        return (
            (self.r2s**2 - self.r2h**2)
            * math.pi
            * self.blockage[1]
            * math.cos(self.beta2 / 180.0 * math.pi)
        )

    @property
    def A_y(self):
        # Effective area at station 3
        return (
            (self.r2s**2 - self.r2h**2)
            * math.pi
            * math.cos(self.beta2 / 180 * math.pi)
            - (self.r2s - self.r2h) * self.blade_e * self.n_blades
        ) * self.blockage[2]

    @property
    def beta2_opt(self):
        return (
            math.atan(self.A_x / self.A_y * math.tan(self.beta2 / 180 * math.pi))
            * 180
            / math.pi
        )

    @property
    def slip(self):
        """Slip according to Wiesner-Busemann"""
        return (
            1
            - (math.cos(self.beta4 / 180 * math.pi)) ** 0.5
            / (self.n_blades + self.n_splits) ** 0.7
        )

    @property
    def eps_limit(self):
        pass

    @property
    def hydraulic_diameter(self):
        la = self.r2h / self.r2s
        Dh = (
            2
            * self.r4
            * (
                1.0
                / (
                    self.n_blades / math.pi / math.cos(self.beta4 / 180 * math.pi)
                    + 2.0 * self.r4 / self.b4
                )
                + self.r2s
                / self.r4
                / (
                    2.0 / (1.0 - la)
                    + 2.0
                    * (self.n_blades)
                    / math.pi
                    / (1 + la)
                    * (
                        math.sqrt(
                            1
                            + (1 + la**2 / 2)
                            * math.tan(self.beta2s / 180 * math.pi) ** 2
                        )
                    )
                )
            )
        )
        Lh = (
            self.r4
            * (1 - self.r2rms * 2 / 0.3048)
            / (math.cos(self.beta4 / 180 * math.pi))
        )
        return Dh, Lh

    @classmethod
    def from_dict(cls, data: dict, blockage: Union[List[float], None] = None):
        """Create a Geometry instance from a geometry structure from MATLAB or
        from a file"""
        safe_names = [f.name for f in fields(cls)]
        d = {}

        if blockage is None and "blockage1" in data:
            blockage = [data[f"blockage{i+1}"] for i in range(5)]

        if blockage is None:
            raise ValueError("Blockage needs to be provided as an argument or in data.")

        for k, v in data.items():
            if k.lower() in safe_names:
                d[k.lower()] = v

        d["blockage"] = blockage
        return cls(**d)
