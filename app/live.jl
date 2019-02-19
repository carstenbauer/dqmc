using Pkg
Pkg.activate(ENV["JULIA_DQMC"])

using Revise, BenchmarkTools
include(joinpath(ENV["JULIA_DQMC"], "src/dqmc_framework.jl"))


if !(@isdefined input)
    if isfile("dqmc.in.xml")
        input = "dqmc.in.xml"
    else
        error("Variable 'input' != path to input.in.xml")
    end
end


p = Params()
p.output_file = "live.out.h5.running"
xml2parameters!(p, input)

mc = DQMC(p)


init!(mc)

display(mc)
nothing