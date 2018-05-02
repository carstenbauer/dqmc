using Plots, DataFrames, StatPlots, NamedTuples, JLD, Query
default(:framestyle, :box)

@load "df.jld" df

sort!(df, cols=[:CB, :B, :L])

pyplot(size=(700,1000))

# @df df plot(:L, [:multiply_B, :wrap_greens, :slice_matrix], group=@NT(CB = :CB, B = :B),
# 	marker=:circle,
# 	xlab="L",
# 	ylab=["multiply_B" "wrap_greens" "slice_matrix"],
# 	layout=grid(3,1, heights=(.33,.33,.33))
# )
# savefig("CB_vs_L.pdf")


# ----------------------------------
#	fixed B, vs L
# ----------------------------------
d = @from i in df begin
	@where i.B == 10
	@select i
	@collect DataFrame
end

@df d plot(:L, [:multiply_B, :wrap_greens, :slice_matrix], group=:CB,
	marker=:circle,
	xlab="L",
	ylab=["multiply_B" "wrap_greens" "slice_matrix"],
	layout=grid(3,1, heights=(.33,.33,.33))
)
savefig("CB_vs_L.pdf")

@df d plot(:L, [:multiply_B, :wrap_greens, :slice_matrix], group=:CB,
	scale=:log10,
	marker=:circle,
	xlab="L",
	ylab=["multiply_B" "wrap_greens" "slice_matrix"],
	layout=grid(3,1, heights=(.33,.33,.33))
)
savefig("CB_vs_L_loglog.pdf")


# ----------------------------------
#	fixed L, vs B
# ----------------------------------

d = @from i in df begin
	@where i.L == 10
	@select i
	@collect DataFrame
end

@df d plot(:B, [:multiply_B, :wrap_greens, :slice_matrix], group=:CB,
	marker=:circle,
	xlab="Beta",
	ylab=["multiply_B" "wrap_greens" "slice_matrix"],
	layout=grid(3,1, heights=(.33,.33,.33))
)
savefig("CB_vs_B.pdf")

@df d plot(:B, [:multiply_B, :wrap_greens, :slice_matrix], group=:CB,
	scale=:log10,
	marker=:circle,
	xlab="Beta",
	ylab=["multiply_B" "wrap_greens" "slice_matrix"],
	layout=grid(3,1, heights=(.33,.33,.33))
)
savefig("CB_vs_B_loglog.pdf")