# Checkerboard: Assaad four site version for square lattice
function build_four_site_hopping_matrix(p::Parameters,l::Lattice,corner::Int,tflv::String="x")
  sites_clockwise = [corner, l.neighbors[1,corner], l.neighbors[1,l.neighbors[2,corner]], l.neighbors[2,corner]]
  shift_sites_clockwise = circshift(sites_clockwise,-1)
  hop = zeros(8)
  if tflv=="y"
    hop = -1 * repeat([l.t.yv, l.t.yh, l.t.yv, l.t.yh], outer=2)
  else
    hop = -1 * repeat([l.t.xv, l.t.xh, l.t.xv, l.t.xh], outer=2)
  end
  return sparse([sites_clockwise; shift_sites_clockwise],[shift_sites_clockwise; sites_clockwise],hop,l.sites,l.sites)
end


function init_checkerboard_matrices(p::Parameters, l::Lattice)
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

  l.chkr_hop_4site = Array{SparseMatrixCSC, 3}(pc,2,2)

  # TODO: Cutoff of numerical zeros necessary. Better construct exponentials explicitly (sinh cosh)
  low_cutoff(X::Array{Float64}, c::Float64) = map(e->abs(e)<abs(c)?0.:e,X)
  for c in 1:pc
    l.chkr_hop_4site[c,1,1] = sparse(low_cutoff(expm(-0.5 * p.delta_tau * full(build_four_site_hopping_matrix(p, l, A_corners[c], "x"))),1e-15)) # eTx_As
    l.chkr_hop_4site[c,1,2] = sparse(low_cutoff(expm(-0.5 * p.delta_tau * full(build_four_site_hopping_matrix(p, l, A_corners[c], "y"))),1e-15)) # eTy_As
    l.chkr_hop_4site[c,2,1] = sparse(low_cutoff(expm(-0.5 * p.delta_tau * full(build_four_site_hopping_matrix(p, l, B_corners[c], "x"))),1e-15)) # eTx_Bs
    l.chkr_hop_4site[c,2,2] = sparse(low_cutoff(expm(-0.5 * p.delta_tau * full(build_four_site_hopping_matrix(p, l, B_corners[c], "y"))),1e-15)) # eTy_Bs
  end

  # Calculate full subgroup hopping matrix exps and compare to exact hopping matrix exp
  eTx_A = full(foldl(*,l.chkr_hop_4site[:,1,1]))
  eTx_B = full(foldl(*,l.chkr_hop_4site[:,2,1]))
  eTy_A = full(foldl(*,l.chkr_hop_4site[:,1,2]))
  eTy_B = full(foldl(*,l.chkr_hop_4site[:,2,2]))

  eT_A = cat([1,2],eTx_A,eTy_A,eTx_A,eTy_A)
  eT_B = cat([1,2],eTx_B,eTy_B,eTx_B,eTy_B)
  eT_A_inv = cat([1,2],eTx_A_inv,eTy_A_inv,eTx_A_inv,eTy_A_inv)
  eT_B_inv = cat([1,2],eTx_B_inv,eTy_B_inv,eTx_B_inv,eTy_B_inv)

  l.chkr_hop = [eT_A, eT_B]
  l.chkr_hop_inv = [eT_A_inv, eT_B_inv]
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


function multiply_four_site_chkr_hops_left!(A::Union{Array, SubArray}, subgroup::Int, tflv::Int)
  # Max explicitly hardcodes the effect of the multiplication (only changing four rows)
  for h in l.chkr_hop_4site[:,subgroup,tflv]
    A[:] = h * A
  end
end

function multiply_four_site_chkr_hops_right!(A::Union{Array, SubArray}, subgroup::Int, tflv::Int)
  # Max explicitly hardcodes the effect of the multiplication (only changing four columns)
  for h in l.chkr_hop_4site[:,subgroup,tflv]
    A[:] = A * h
  end
end

function block(p::Parameters, l::Lattice, M::Matrix, row::Int, col::Int)
  return view(M, ((row-1)*l.sites+1):row*l.sites, ((col-1)*l.sites+1):col*l.sites )
end

function multiply_slice_matrix_left!(p::Parameters, l::Lattice, slice::Int, A::Matrix{Complex128})
  # e^(- dtau TA/2)
  for col in 1:p.flv
      multiply_four_site_chkr_hops_left!(block(p,l, A, 1, col), 1, 1)
      multiply_four_site_chkr_hops_left!(block(p,l, A, 2, col), 1, 2)
      multiply_four_site_chkr_hops_left!(block(p,l, A, 3, col), 1, 1)
      multiply_four_site_chkr_hops_left!(block(p,l, A, 4, col), 1, 2)
  end

  # e^(- dtau TB/2)
  for col in 1:p.flv
      multiply_four_site_chkr_hops_left!(block(p,l, A, 1, col), 2, 1)
      multiply_four_site_chkr_hops_left!(block(p,l, A, 2, col), 2, 2)
      multiply_four_site_chkr_hops_left!(block(p,l, A, 3, col), 2, 1)
      multiply_four_site_chkr_hops_left!(block(p,l, A, 4, col), 2, 2)
  end

  # e^(- dtau mu) e^(- dtau V) e^(- dtau mu)
  A[:] = l.chkr_mu * interaction_matrix_exp(p, l, slice) * l.chkr_mu * A

  # e^(- dtau TB/2)
  for col in 1:p.flv
      multiply_four_site_chkr_hops_left!(block(p,l, A, 1, col), 2, 1)
      multiply_four_site_chkr_hops_left!(block(p,l, A, 2, col), 2, 2)
      multiply_four_site_chkr_hops_left!(block(p,l, A, 3, col), 2, 1)
      multiply_four_site_chkr_hops_left!(block(p,l, A, 4, col), 2, 2)
  end

  # e^(- dtau TA/2)
  for col in 1:p.flv
      multiply_four_site_chkr_hops_left!(block(p,l, A, 1, col), 1, 1)
      multiply_four_site_chkr_hops_left!(block(p,l, A, 2, col), 1, 2)
      multiply_four_site_chkr_hops_left!(block(p,l, A, 3, col), 1, 1)
      multiply_four_site_chkr_hops_left!(block(p,l, A, 4, col), 1, 2)
  end
  nothing
end
