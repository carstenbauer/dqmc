using Plots; pyplot();
using DataFrames, StatPlots, BenchmarkTools, NamedTuples
default(:framestyle, :box)

df = DataFrame(L = Int[], B = Int[], qr = Float64[], svd = Float64[])
input = "test.in.xml"

for (l, L) in enumerate([2,4,6,8,10])
	for (b, B) in enumerate([2])
		xml2parameters!(p, input)
		p.L = L
		p.lattice_file = "C:\\Users\\carsten\\Desktop\\sciebo\\lattices\\square_L_$(L)_W_$(L).xml"
		p.slices = Int(B/p.delta_tau)
		deduce_remaining_parameters(p)
		mc = DQMC(p)
		init!(mc)

		tqr = @belapsed calculate_greens_and_logdet($mc, 1, 1)
		tsvd = @belapsed calculate_greens_and_logdet_udv($mc, 1, 1)

		push!(df, [L, B, tqr, tsvd])
	end
end

plot(df[:L], [df[:qr], df[:svd]],
	marker=:circle,
	group=@NT(B = df[:B]),
	lab=["qr" "svd"],
	xlab="linear system size",
	ylab="elapsed time GF calculation",
	# scale=:log10
)
savefig("qr_vs_svd.pdf")

plot(df[:L], [df[:qr], df[:svd]],
	marker=:circle,
	group=@NT(B = df[:B]),
	lab=["qr" "svd"],
	xlab="linear system size",
	ylab="elapsed time GF calculation",
	scale=:log10
)
savefig("qr_vs_svd_loglog.pdf")