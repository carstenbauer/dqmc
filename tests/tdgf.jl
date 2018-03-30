using Plots; pyplot();

default(:framestyle, :box)

n = Int(p.slices)

tdgfs_naive = zeros(n);
tdgfs = zeros(n);
for slice in 1:n
	tdgfcx_naive = calculate_tdgf_naive(s,p,l,slice)[1]
	tdgfcx = calculate_tdgf(s,p,l,slice)[1]
	# @assert isreal(tdgfcx_naive)
	# @assert isreal(tdgfcx)
	tdgfs_naive[slice] = real(tdgfcx_naive)
	tdgfs[slice] = real(tdgfcx)
end

tdgfs[tdgfs.<0] = NaN

plot((1:n).*p.delta_tau, [tdgfs, tdgfs_naive], 
	xlab="\$ \\tau \$", 
	ylab="time-displaced Green's function", 
	# yscale=:log10,
	lab=["stabilized" "naive"],
    marker=:circle
)
savefig("tdgf_free_dqmc.pdf")