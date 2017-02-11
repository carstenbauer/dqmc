# Checkerboard: Assaad four site version for square lattice
function build_four_site_hopping_matrix(p::Parameters,l::Lattice,corner::Int,flv::String="x")
  sites_clockwise = [corner, l.neighbors[1,corner], l.neighbors[1,l.neighbors[2,corner]], l.neighbors[2,corner]]
  shift_sites_clockwise = circshift(sites_clockwise,-1)
  hop = zeros(8)
  if flv=="y"
    hop = -1 * repeat([l.t.yv, l.t.yh, l.t.yv, l.t.yh], outer=2)
  else
    hop = -1 * repeat([l.t.xv, l.t.xh, l.t.xv, l.t.xh], outer=2)
  end
  return sparse([sites_clockwise; shift_sites_clockwise],[shift_sites_clockwise; sites_clockwise],hop,l.sites,l.sites)
end

function init_checkerboard_matrices(p::Parameters,l::Lattice)
  pc = Int(floor(l.sites/4))

  A_corners = zeros(Int, pc)
  idx = 1
  tolinidx = reshape(collect(1:l.sites),l.L,l.L)
  for x in 1:2:l.L
      for y in 1:2:l.L
          A_corners[idx] = tolinidx[y,x]
          idx += 1
      end
  end
  B_corners = l.neighbors[1,l.neighbors[2,A_corners]]

  Tx_As = Array{SparseMatrixCSC,1}(pc)
  Tx_Bs = Array{SparseMatrixCSC,1}(pc)
  Ty_As = Array{SparseMatrixCSC,1}(pc)
  Ty_Bs = Array{SparseMatrixCSC,1}(pc)

  for c in 1:length(A_corners)
    Tx_As[c] = build_four_site_hopping_matrix(p, l, A_corners[c], "x")
    Ty_As[c] = build_four_site_hopping_matrix(p, l, A_corners[c], "y")
    Tx_Bs[c] = build_four_site_hopping_matrix(p, l, B_corners[c], "x")
    Ty_Bs[c] = build_four_site_hopping_matrix(p, l, B_corners[c], "y")
  end

  Tx_A = sum(Tx_As)
  Ty_A = sum(Ty_As)
  Tx_B = sum(Tx_Bs)
  Ty_B = sum(Ty_Bs)

  # To remove numerical quasi zeros, which should be (Mathematica) exactly zero
  low_cutoff(X::Array, c::Float64) = map(e->abs(e)<abs(c)?0.:e,X)
  eTx_A_minus = sparse(low_cutoff(expm(-0.5 * p.delta_tau * full(Tx_A)), 1e-15))
  eTy_A_minus = sparse(low_cutoff(expm(-0.5 * p.delta_tau * full(Ty_A)), 1e-15))
  eTx_B_minus = sparse(low_cutoff(expm(-0.5 * p.delta_tau * full(Tx_B)), 1e-15))
  eTy_B_minus = sparse(low_cutoff(expm(-0.5 * p.delta_tau * full(Ty_B)), 1e-15))
  eTx_A_plus = sparse(low_cutoff(expm(0.5 * p.delta_tau * full(Tx_A)), 1e-15))
  eTy_A_plus = sparse(low_cutoff(expm(0.5 * p.delta_tau * full(Ty_A)), 1e-15))
  eTx_B_plus = sparse(low_cutoff(expm(0.5 * p.delta_tau * full(Tx_B)), 1e-15))
  eTy_B_plus = sparse(low_cutoff(expm(0.5 * p.delta_tau * full(Ty_B)), 1e-15))

  # chemical potential
  # eTmu_minus = spdiagm(fill(exp(-0.5 * p.delta_tau * -p.mu), l.sites))
  # eTmu_plus = spdiagm(fill(exp(0.5 * p.delta_tau * -p.mu), l.sites))

  # not-sparse matrix eTx_A_minus * eTx_B_minus * eTmu_minus is equal to eTx_minus above (non-chkr hopping matrix)
  # individual eTx_A_minus have sparsity ~0.75

  eT_A_minus = cat([1,2],eTx_A_minus,eTy_A_minus,eTx_A_minus,eTy_A_minus) # sparsity ~ 93%
  eT_B_minus = cat([1,2],eTx_B_minus,eTy_B_minus,eTx_B_minus,eTy_B_minus)
  eT_A_plus = cat([1,2],eTx_A_plus,eTy_A_plus,eTx_A_plus,eTy_A_plus)
  eT_B_plus = cat([1,2],eTx_B_plus,eTy_B_plus,eTx_B_plus,eTy_B_plus)

  l.chkr_hop = [eT_A_minus, eT_B_minus]
  l.chkr_hop_inv = [eT_A_plus, eT_B_plus]
  l.chkr_mu = spdiagm(fill(exp(-0.5 * p.delta_tau * -p.mu), p.flv * l.sites))
  l.chkr_mu_inv = spdiagm(fill(exp(0.5 * p.delta_tau * -p.mu), p.flv * l.sites))

  hop_mat_exp_chkr = l.chkr_hop[1] * l.chkr_hop[2] * l.chkr_mu
  println("Checkerboard - exact (abs):\t\t", maximum(absdiff(l.hopping_matrix_exp,hop_mat_exp_chkr)))
  println("Checkerboard - exact (rel):\t\t", maximum(reldiff(l.hopping_matrix_exp,hop_mat_exp_chkr)))
end


function slice_matrix(p::Parameters, l::Lattice, slice::Int, pref::Float64=1.)

  M = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)

  if pref > 0

    for h in l.chkr_hop
      M = h * M
    end

    M = l.chkr_mu * M
    M = interaction_matrix_exp(p, l, slice) * M
    M = l.chkr_mu * M

    for h in reverse(l.chkr_hop)
      M = h * M
    end
  else

    for h in l.chkr_hop_inv
      M = h * M
    end

    M = l.chkr_mu_inv * M
    M = interaction_matrix_exp(p, l, slice, -1.) * M
    M = l.chkr_mu_inv * M

    for h in reverse(l.chkr_hop_inv)
      M = h * M
    end
  end
  return M
end
