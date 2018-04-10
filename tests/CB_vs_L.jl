using Plots; pyplot()
using DataFrames, StatPlots
default(:framestyle, :box)

input = "CB_vs_L.in.xml"
include("../dqmc_framework.jl")

p = Params()
p.output_file = "CB_vs_L.out.h5.running"
xml2parameters!(p, input)

for (l,L) in enumerate([2,4,6,8])
	p.L = L
	p.lattice_file = ""
	mc = DQMC(p)
	init!(mc)

	t = @elapsed 
end