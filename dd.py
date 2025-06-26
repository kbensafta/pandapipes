##
import pandapipes as pp
from pandapipes.plotting import simple_plot
net = pp.create_empty_network(fluid="lgas") # create an empty network

junction1 = pp.create_junction(net, pn_bar=1.0, tfluid_k=293.15, name="Connection to External Grid")
junction2 = pp.create_junction(net, pn_bar=1.0, tfluid_k=293.15, name="Junction 2")
junction3 = pp.create_junction(net, pn_bar=1.0, tfluid_k=293.15, name="Junction 3")
# junction4 = pp.create_junction(net, pn_bar=1.0, tfluid_k=293.15, name="Junction 4")
# junction5 = pp.create_junction(net, pn_bar=1.0, tfluid_k=293.15, name="Junction 5")
# junction6 = pp.create_junction(net, pn_bar=1.0, tfluid_k=293.15, name="Junction 6")

medium_pressure_grid = pp.create_ext_grid(net, junction=junction1, p_bar=1.1, t_k=293.15, name="Grid Connection")

pipe1 = pp.create_pipe_from_parameters(net, from_junction=junction1, to_junction=junction2, length_km=10, diameter_m=0.3, name="Pipe 1")
pipe2 = pp.create_pipe_from_parameters(net, from_junction=junction2, to_junction=junction3, length_km=2, diameter_m=0.4, name="Pipe 2")
# pipe3 = pp.create_pipe_from_parameters(net, from_junction=junction2, to_junction=junction4, length_km=2.5, diameter_m=0.3, name="Pipe 3")
# pipe4 = pp.create_pipe_from_parameters(net, from_junction=junction3, to_junction=junction5, length_km=1, diameter_m=0.3, name="Pipe 4")
# pipe5 = pp.create_pipe_from_parameters(net, from_junction=junction4, to_junction=junction6, length_km=1, diameter_m=0.3, name="Pipe 5")

#pp.create_valve(from_junction=junction1, to_junction=junction2)

#valve = pp.create_valve(net, from_junction=junction5, to_junction=junction6, diameter_m=0.3, opened=True)
sink = pp.create_sink(net, junction=junction3, mdot_kg_per_s=0.545, name="Sink 1")
# source = pp.create_source(net, junction=junction3, mdot_kg_per_s=0.234, name="Source 1")

#simple_plot(net, plot_sinks=True, plot_sources=True, plot_valves=True)

#net["user_pf_options"]
##
pp.pipeflow(net, use_numba=True)
net["user_pf_options"]

