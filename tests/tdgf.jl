using Plots; pyplot();

default(:framestyle, :box)

n = Int(p.slices)

tdgfs_naive = zeros(n);
tdgfs = zeros(n);
for slice in 1:n
	tdgfcx_naive = mean(diag(calculate_tdgf_naive(s,p,l,slice)))
	tdgfcx = mean(diag(calculate_tdgf(s,p,l,slice)))
	# @assert isreal(tdgfcx_naive)
	# @assert isreal(tdgfcx)
	tdgfs_naive[slice] = real(tdgfcx_naive)
	tdgfs[slice] = real(tdgfcx)
end

tdgfs[tdgfs.<0] = NaN

# equal times gf for comparison at tau=0
g, = calculate_greens_and_logdet(s,p,l,1)
effective_greens2greens_no_chkr!(p,l,g)
gii = mean(diag(g))

# plot dqmc
plot((1:n).*p.delta_tau, [tdgfs, tdgfs_naive], 
	xlab="\$ \\tau \$", 
	ylab="\$ \\langle c_i(\\tau) c_i(0)^\\dagger \\rangle \$", 
	# yscale=:log10,
	lab=["stabilized" "naive"],
    marker=:circle,
    ylim=(-.25,.75)
)
pl = plot!([p.beta/2], seriestype=:vline, lab="\$ \\beta/2 \$", color=:grey, line=:dash)
savefig("tdgf_free_dqmc.pdf")

# plot dqmc together with ed
using HDF5
tdgfs_ed = h5read("../tdgfed.jld", "tdgfiivstau")
pl = plot((1:n).*p.delta_tau, [tdgfs, tdgfs_naive], 
	xlab="\$ \\tau \$", 
	ylab="\$ \\langle c_i(\\tau) c_i(0)^\\dagger \\rangle \$", 
	# yscale=:log10,
	lab=["stabilized" "naive"],
    marker=:circle,
    ylim=(-.25,.75)
)
pl = plot!(linspace(0.1,p.beta,80), tdgfs_ed, lab="ed")
pl = plot!([0.0, p.beta], [gii, gii], 
	color=:black,
	lab="\$ \\langle c_i c_i^\\dagger \\rangle \$",
	seriestype=:scatter,
	marker=:circle
)
pl = plot!([p.beta/2], seriestype=:vline, lab="\$ \\beta/2 \$", color=:grey, line=:dash)
savefig(pl, "tdgf_free.pdf")

# plot dqmc together with ed (zoomed in at beta/2)
using HDF5
tdgfs_ed = h5read("../tdgfed.jld", "tdgfiivstau")
pl = plot((1:n).*p.delta_tau, [tdgfs, tdgfs_naive], 
	xlab="\$ \\tau \$", 
	ylab="\$ \\langle c_i(\\tau) c_i(0)^\\dagger \\rangle \$", 
	# yscale=:log10,
	lab=["stabilized" "naive"],
    marker=:circle,
    xlim=(3,5),
    ylim=(0.02,0.06)
)
pl = plot!(linspace(0.1,p.beta,80), tdgfs_ed, lab="ed")
pl = plot!([p.beta/2], seriestype=:vline, lab="\$ \\beta/2 \$", color=:grey, line=:dash)
savefig(pl, "tdgf_free_zoomed.pdf")








tdgf_ed = h5read("../tdgfed.jld", "tdgf")
tdgf = calculate_tdgf(s,p,l,40)
tdgf_naive = calculate_tdgf_naive(s,p,l,40)
@show maximum(absdiff(tdgf_ed[1:8,1:8], tdgf)) # 4.89e-5
@show maximum(absdiff(tdgf_ed[1:8,1:8], tdgf_naive)) # 89.5073651556123