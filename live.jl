isdefined(:input) || error("Variable 'input' != path to input.in.xml")

# using Revise
include("dqmc_framework.jl")
using BenchmarkTools

p = Params()
p.output_file = "live.out.h5.running"
xml2parameters!(p, input)

mc = DQMC(p)
init!(mc)

# const l = mc.l;
# const s = mc.s;
display(mc)
nothing