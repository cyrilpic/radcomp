try:
    import CoolProp as CP
    from CoolProp.Plots import PropertyPlot, StateContainer
except ImportError:
    CP, PropertyPlot, StateContainer = None, None, None

from .compressor import Compressor


def plot_compressor_cycle(comp: Compressor, plot_type="Ts") -> PropertyPlot:
    plot = PropertyPlot(comp.op.fld.name.capitalize(), plot_type)
    plot.calc_isolines(CP.iQ, num=11)
    plot.calc_isolines(CP.iP, num=15)

    states = StateContainer()

    states[0, "H"] = comp.in_.total.H
    states[0, "S"] = comp.in_.total.S
    states[0, "T"] = comp.in_.total.T
    states[0, "P"] = comp.in_.total.P

    states[1, "H"] = comp.ind.out.total.H
    states[1, "S"] = comp.ind.out.total.S
    states[1, "T"] = comp.ind.out.total.T
    states[1, "P"] = comp.ind.out.total.P

    states[2, "H"] = comp.imp.out.total.H
    states[2, "S"] = comp.imp.out.total.S
    states[2, "T"] = comp.imp.out.total.T
    states[2, "P"] = comp.imp.out.total.P

    states[3, "H"] = comp.out.total.H
    states[3, "S"] = comp.out.total.S
    states[3, "T"] = comp.out.total.T
    states[3, "P"] = comp.out.total.P

    plot.draw_process(states)
    return plot
