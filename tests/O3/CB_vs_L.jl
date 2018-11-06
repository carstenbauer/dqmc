using Plots;
using DataFrames, StatPlots, BenchmarkTools, NamedTuples, JLD, Query
default(:framestyle, :box)

input = "CB_vs_L.in.xml"
include("../../live.jl")

if !isfile("df.jld")
	global df = DataFrame(L=Int[], B=Int[], CB=String[], 
		multiply_B=Float64[], slice_matrix_chain=Float64[])
else
	global df = load("df.jld", "df")
end

function measure(mc::AbstractDQMC,p::Params,df::DataFrame, 
					B::Int, L::Int, CB::String, cb::Bool)
	p.L = L
	p.lattice_file = "/home/bauer/lattices/square_L_$(p.L)_W_$(p.L).xml"
	p.chkr = cb
	p.slices = B*10
	p.beta = B
	mc = DQMC(p)
	init!(mc)
	M = copy(mc.s.greens)

	tB = @belapsed multiply_slice_matrix_left!($mc, 1, $M)
	tchain = @belapsed calculate_slice_matrix_chain_qr(mc, 1, p.slices, 10)
	# tgreens = @belapsed measure_greens($mc)
	# tslicem = @belapsed slice_matrix($mc, 1)
	# tstack = @belapsed build_stack($mc)
	# twrap = @belapsed wrap_greens!($mc, $mc.s.greens, $mc.s.current_slice, $mc.s.direction)

	push!(df, [L, Int(p.beta), string(cbtype(mc)), tB, tchain])
	nothing
end


for cb in [false, true]
	for (Bi, B) in enumerate([2,4,6,8,10,20])
		for (Li, L) in enumerate(2:24)
			CB = "CBFalse"
			cb && (CB = iseven(L) ? "CBAssaad" : "CBGeneric")

			println("------------------")
			@show L
			@show B
			@show CB
			flush(stdout)

			# check if we have that already
			x = @from i in df begin
			    @where i.L == L && i.B == B && i.CB == CB
			    @select i
			end
			isempty(x) || continue

			measure(mc,p,df,B,L,CB,cb)

			println("Dumping...")
			flush(stdout)
			save("df.jld", "df", df)
		end
	end
end

sort!(df, cols=[:CB, :B, :L])
@save "df.jld" df
println("Done.!")