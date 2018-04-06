isdefined(:input) || error("Variable 'input' != path to input.in.xml")

include("dqmc_framework.jl")

p = Parameters()
p.output_file = "live.out.h5.running"
xml2parameters!(p, input)

mc = DQMC(p)
init!(mc)