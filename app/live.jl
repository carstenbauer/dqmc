using Pkg
Pkg.activate(ENV["JULIA_DQMC"])

using Revise, BenchmarkTools
include(joinpath(ENV["JULIA_DQMC"], "src/dqmc_framework.jl"))


if !(@isdefined infile)
    if isfile("dqmc.in.xml")
        global infile = "dqmc.in.xml"
    else
        error("Variable 'infile' != path to infile.in.xml")
    end
end


p = Params()
p.output_file = "live.out.h5.running"
xml2parameters!(p, infile)

mc = DQMC(p)


init!(mc)

display(mc)
nothing
