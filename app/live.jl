if !(@isdefined input)
    if isfile("dqmc.in.xml")
        input = "dqmc.in.xml"
    else
        error("Variable 'input' != path to input.in.xml")
    end
end

using Revise, BenchmarkTools
include("../src/dqmc_framework.jl")

p = Params()
p.output_file = "live.out.h5.running"
xml2parameters!(p, input)

mc = DQMC(p)


init!(mc)

# const l = mc.l;
# const s = mc.s;
meas = mc.s.meas
display(mc)
nothing