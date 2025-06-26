from matplotlib import pyplot as plt
from pandapower.plotting import draw_collections

import pandapipes.plotting as plot
import pandapipes as ppi
from pandapipes.plotting import simple_plot, create_simple_collections


def create_t_net(nbr_junctions=4, nbr_pipes=3):
    net = ppi.create_empty_network()

    junctions = ppi.create_junctions(net, nr_junctions=nbr_junctions, pn_bar=nbr_junctions*[10], tfluid_k=nbr_junctions*[283.15])
    from_junct = junctions[0], junctions[1], junctions[2]
    to_junct = junctions[2], junctions[2], junctions[3]

    ppi.create_pipes_from_parameters(net, from_junctions=from_junct, to_junctions=to_junct, diameter_m=nbr_pipes*[0.1],
                                     length_km=nbr_pipes*[5])

    ppi.create_ext_grid(net, junction=junctions[0], fluid='hgas', p_bar=10)
    ppi.create_source(net, junction=junctions[1], fluid='hydrogen', mdot_kg_per_s=0.02)
    ppi.create_sink(net, junction=junctions[3], mdot_kg_per_s=0.04)

    simple_plot(net, plot_sinks=True, plot_sources=True)

    ppi.pipeflow(net)

    return net

t_net = create_t_net()


def create_tt_net(nbr_junctions=6, nbr_pipes=5):
    net = ppi.create_empty_network()

    junctions = ppi.create_junctions(net, nr_junctions=nbr_junctions, pn_bar=nbr_junctions * [10],
                                     tfluid_k=nbr_junctions * [283.15])
    from_junct = junctions[0], junctions[1], junctions[2], junctions[1], junctions[2]
    to_junct = junctions[1], junctions[2], junctions[3], junctions[4], junctions[5]

    ppi.create_pipes_from_parameters(net, from_junctions=from_junct, to_junctions=to_junct,
                                     diameter_m=nbr_pipes * [0.1],
                                     length_km=nbr_pipes * [5])

    ppi.create_ext_grid(net, junction=junctions[0], fluid='hgas', p_bar=10)
    ppi.create_source(net, junction=junctions[4], fluid='hydrogen', mdot_kg_per_s=0.02)
    ppi.create_source(net, junction=junctions[5], fluid='hydrogen', mdot_kg_per_s=0.02)
    ppi.create_sink(net, junction=junctions[3], mdot_kg_per_s=0.06)

    simple_plot(net, plot_sinks=True, plot_sources=True)

    ppi.pipeflow(net)

    return net

tt_net = create_tt_net()

def create_ttt_net(nbr_junctions=8, nbr_pipes=7):
    net = ppi.create_empty_network()

    junctions = ppi.create_junctions(net, nr_junctions=nbr_junctions, pn_bar=nbr_junctions * [10],
                                     tfluid_k=nbr_junctions * [283.15])
    from_junct = junctions[0], junctions[1], junctions[2], junctions[3], junctions[1], junctions[2], junctions[3]
    to_junct = junctions[1], junctions[2], junctions[3], junctions[4], junctions[5], junctions[6],junctions[7]

    ppi.create_pipes_from_parameters(net, from_junctions=from_junct, to_junctions=to_junct,
                                     diameter_m=nbr_pipes * [0.1],
                                     length_km=nbr_pipes * [5])

    ppi.create_ext_grid(net, junction=junctions[0], fluid='hgas', p_bar=10)
    ppi.create_source(net, junction=junctions[5], fluid='hydrogen', mdot_kg_per_s=0.02)
    ppi.create_source(net, junction=junctions[6], fluid='hydrogen', mdot_kg_per_s=0.02)
    ppi.create_source(net, junction=junctions[7], fluid='hydrogen', mdot_kg_per_s=0.02)
    ppi.create_sink(net, junction=junctions[4], mdot_kg_per_s=0.08)

    simple_plot(net, plot_sinks=True, plot_sources=True)

    ppi.pipeflow(net)

    return net

ttt_net = create_ttt_net()


def create_ttt_vx_net(nbr_junctions=8, nbr_pipes=7):
    net = ppi.create_empty_network()

    junctions = ppi.create_junctions(net, nr_junctions=nbr_junctions, pn_bar=nbr_junctions * [10],
                                     tfluid_k=nbr_junctions * [283.15])
    from_junct = junctions[0], junctions[1], junctions[2], junctions[3], junctions[1], junctions[2], junctions[3]
    to_junct = junctions[1], junctions[2], junctions[3], junctions[4], junctions[5], junctions[6],junctions[7]

    ppi.create_pipes_from_parameters(net, from_junctions=from_junct, to_junctions=to_junct,
                                     diameter_m=nbr_pipes * [0.1],
                                     length_km=nbr_pipes * [5])

    ppi.create_ext_grid(net, junction=junctions[0], fluid='hgas', p_bar=10)
    ppi.create_source(net, junction=junctions[5], fluid='hydrogen', mdot_kg_per_s=0.02)
    ppi.create_sink(net, junction=junctions[6], mdot_kg_per_s=0.02)
    ppi.create_source(net, junction=junctions[7], fluid='hydrogen', mdot_kg_per_s=0.02)
    ppi.create_sink(net, junction=junctions[4], mdot_kg_per_s=0.08)

    simple_plot(net, plot_sinks=True, plot_sources=True)

    ppi.pipeflow(net)

    return net


ttt_vx_net = create_ttt_vx_net()


def create_ttt_vy_net(nbr_junctions=8, nbr_pipes=7):
    net = ppi.create_empty_network()

    junctions = ppi.create_junctions(net, nr_junctions=nbr_junctions, pn_bar=nbr_junctions * [10],
                                     tfluid_k=nbr_junctions * [283.15])
    from_junct = junctions[0], junctions[1], junctions[2], junctions[3], junctions[1], junctions[2], junctions[3]
    to_junct = junctions[1], junctions[2], junctions[3], junctions[4], junctions[5], junctions[6],junctions[7]

    ppi.create_pipes_from_parameters(net, from_junctions=from_junct, to_junctions=to_junct,
                                     diameter_m=nbr_pipes * [0.1],
                                     length_km=nbr_pipes * [5])

    ppi.create_ext_grid(net, junction=junctions[0], fluid='hgas', p_bar=10)
    ppi.create_source(net, junction=junctions[5], fluid='hydrogen', mdot_kg_per_s=0.02)
    ppi.create_sink(net, junction=junctions[6], mdot_kg_per_s=0.02)
    ppi.create_source(net, junction=junctions[7], fluid='hydrogen', mdot_kg_per_s=0.02)
    ppi.create_sink(net, junction=junctions[4], mdot_kg_per_s=0.08)

    ppi.create_p

    simple_plot(net, plot_sinks=True, plot_sources=True)

    ppi.pipeflow(net)

    return net


ttt_vx_net = create_ttt_vy_net()

