using Plots;
using DataFrames, StatPlots, BenchmarkTools, NamedTuples
default(:framestyle, :box)

input = "CB_vs_L.in.xml"
include("../live.jl")

df = DataFrame(L=Int[], CB=String[], 
	multiply_B=Float64[], greens=Float64[], build_stack=Float64[])

for (Li,L) in enumerate([2,3,4,5])
	println("------------------")
	@show L

	p.L = L
	p.lattice_file = "C:\\Users\\carsten\\Desktop\\sciebo\\lattices\\square_L_$(p.L)_W_$(p.L).xml"
	mc = DQMC(p)
	init!(mc)
	global g = copy(mc.s.greens)
	M = copy(g)

	tB = @belapsed multiply_slice_matrix_left!($mc, 1, $M) setup=(M=copy(g))
	tgreens = @belapsed measure_greens($mc)
	tstack = @belapsed build_stack($mc)

	push!(df, [L, string(cbtype(mc)), tB, tgreens, tstack])
end

pyplot(size=(800,800))

@df df plot(:L, [:multiply_B, :greens], group=(:CB),
	marker=:circle,
	xlab="L",
	ylab="elapsed time",
	layout=grid(2,1, heights=(.5,.5))
)