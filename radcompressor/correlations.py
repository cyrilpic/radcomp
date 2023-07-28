import math

from scipy import optimize


def moody(Re: float, r: float) -> float:
    """Caluclate Moody's coefficient"""
    if Re < 2300.0:
        return 64 / Re

    def colebrook(x: float) -> float:
        return -2 * math.log10(r / 3.72 + 2.51 / Re / x**0.5) - 1 / x**0.5

    return optimize.fsolve(colebrook, 0.02)[0]
