import pytest
from radcompressor.correlations import moody


def test_laminar_flow():
    # Test that the function returns the correct value for laminar flow
    Re = 1000.0
    r = 0.01
    assert moody(Re, r) == pytest.approx(64 / Re, rel=1e-3)


def test_turbulent_flow():
    # Test that the function returns the correct value for turbulent flow
    Re = 5000.0
    r = 0.01
    assert moody(Re, r) == pytest.approx(0.0472, rel=1e-3)
