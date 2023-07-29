from radcompressor.thermo import CoolPropFluid


def test_coolprop_fluid():
    # Test that the CoolPropFluid class returns the correct properties
    water = CoolPropFluid("water")
    assert water.T_crit == 647.096
    assert water.P_crit == 22064000.0

    tp = water.thermo_prop("TQ", 290, 0)
    assert tp.D == 998.7578446208877
