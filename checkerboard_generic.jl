if !isdefined(:HoppingType)
  global const HoppingType = Complex128; # assume O(2) or O(3)
  warn("HoppingType wasn't set on loading checkerboard.jl")
  println("HoppingType = ", HoppingType)
end

















"""
Checkerboard initialization: Assaad four site version for square lattice
"""
function find_four_site_hopping_corners(l::Lattice)
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
  return A_corners, B_corners
end

function build_four_site_hopping_matrix_exp(p::Parameters,l::Lattice, corners::Tuple{Array{Int64,1},Array{Int64,1}}, prefac::Float64=0.5)

  pc = Int(floor(l.sites/4))
  chkr_hop_4site = Array{SparseMatrixCSC, 3}(pc,2,2) # i, group (A, B), hopping flavor (tx, ty)
  chkr_hop_4site_inv = Array{SparseMatrixCSC, 3}(pc,2,2)

  # build explicitly based on analytic form (only without mag field)
  for tflv in 1:2
    for g in 1:2
      for c in 1:pc

        chkr_hop_4site[c,g,tflv] = speye(l.sites)
        chkr_hop_4site_inv[c,g,tflv] = speye(l.sites)

        # Tijmn where i ist bottom left and i<j<m<n.
        i = corners[g][c]
        j = l.neighbors[1,i]
        m = l.neighbors[2,i]
        n = l.neighbors[2,j]

        inds = sort!([corners[g][c], l.neighbors[1,i], l.neighbors[2,i], l.neighbors[2,j]])

        th = l.t[1,tflv]
        tv = l.t[2,tflv]
        fac = -prefac * p.delta_tau

        chkr_hop_4site[c,g,tflv][i,i] = chkr_hop_4site[c,g,tflv][j,j] = chkr_hop_4site[c,g,tflv][m,m] = chkr_hop_4site[c,g,tflv][n,n] = cosh(fac * -th) * cosh(fac * -tv) #
        chkr_hop_4site[c,g,tflv][i,n] = chkr_hop_4site[c,g,tflv][j,m] = chkr_hop_4site[c,g,tflv][m,j] = chkr_hop_4site[c,g,tflv][n,i] = sinh(fac * -th) * sinh(fac * -tv)
        chkr_hop_4site[c,g,tflv][i,j] = chkr_hop_4site[c,g,tflv][j,i] = chkr_hop_4site[c,g,tflv][m,n] = chkr_hop_4site[c,g,tflv][n,m] = cosh(fac * -th) * sinh(fac * -tv) #
        chkr_hop_4site[c,g,tflv][i,m] = chkr_hop_4site[c,g,tflv][j,n] = chkr_hop_4site[c,g,tflv][m,i] = chkr_hop_4site[c,g,tflv][n,j] = sinh(fac * -th) * cosh(fac * -tv)

        chkr_hop_4site_inv[c,g,tflv][i,i] = chkr_hop_4site_inv[c,g,tflv][j,j] = chkr_hop_4site_inv[c,g,tflv][m,m] = chkr_hop_4site_inv[c,g,tflv][n,n] = cosh(-fac * -th) * cosh(-fac * -tv)
        chkr_hop_4site_inv[c,g,tflv][i,n] = chkr_hop_4site_inv[c,g,tflv][j,m] = chkr_hop_4site_inv[c,g,tflv][m,j] = chkr_hop_4site_inv[c,g,tflv][n,i] = sinh(-fac * -th) * sinh(-fac * -tv)
        chkr_hop_4site_inv[c,g,tflv][i,j] = chkr_hop_4site_inv[c,g,tflv][j,i] = chkr_hop_4site_inv[c,g,tflv][m,n] = chkr_hop_4site_inv[c,g,tflv][n,m] = cosh(-fac * -th) * sinh(-fac * -tv)
        chkr_hop_4site_inv[c,g,tflv][i,m] = chkr_hop_4site_inv[c,g,tflv][j,n] = chkr_hop_4site_inv[c,g,tflv][m,i] = chkr_hop_4site_inv[c,g,tflv][n,j] = sinh(-fac * -th) * cosh(-fac * -tv)

      end
    end
  end

  return chkr_hop_4site, chkr_hop_4site_inv
end

function init_checkerboard_matrices(p::Parameters, l::Lattice)

  println("Initializing hopping exponentials (Checkerboard)")
  pc = Int(floor(l.sites/4))

  corners = find_four_site_hopping_corners(l)

  chkr_hop_4site_half, chkr_hop_4site_half_inv = build_four_site_hopping_matrix_exp(p,l, corners, 0.5)
  chkr_hop_4site, chkr_hop_4site_inv = build_four_site_hopping_matrix_exp(p,l, corners, 1.)

  eTx_A_half = foldl(*,chkr_hop_4site_half[:,1,1])
  eTx_B_half = foldl(*,chkr_hop_4site_half[:,2,1])
  eTy_A_half = foldl(*,chkr_hop_4site_half[:,1,2])
  eTy_B_half = foldl(*,chkr_hop_4site_half[:,2,2])
  eTx_A = foldl(*,chkr_hop_4site[:,1,1])
  eTx_B = foldl(*,chkr_hop_4site[:,2,1])
  eTy_A = foldl(*,chkr_hop_4site[:,1,2])
  eTy_B = foldl(*,chkr_hop_4site[:,2,2])

  eTx_A_half_inv = foldl(*,chkr_hop_4site_half_inv[:,1,1])
  eTx_B_half_inv = foldl(*,chkr_hop_4site_half_inv[:,2,1])
  eTy_A_half_inv = foldl(*,chkr_hop_4site_half_inv[:,1,2])
  eTy_B_half_inv = foldl(*,chkr_hop_4site_half_inv[:,2,2])
  eTx_A_inv = foldl(*,chkr_hop_4site_inv[:,1,1])
  eTx_B_inv = foldl(*,chkr_hop_4site_inv[:,2,1])
  eTy_A_inv = foldl(*,chkr_hop_4site_inv[:,1,2])
  eTy_B_inv = foldl(*,chkr_hop_4site_inv[:,2,2])

  if p.opdim == 3
    eT_A_half = cat([1,2],eTx_A_half,eTy_A_half,eTx_A_half,eTy_A_half)
    eT_B_half = cat([1,2],eTx_B_half,eTy_B_half,eTx_B_half,eTy_B_half)
    eT_A_half_inv = cat([1,2],eTx_A_half_inv,eTy_A_half_inv,eTx_A_half_inv,eTy_A_half_inv)
    eT_B_half_inv = cat([1,2],eTx_B_half_inv,eTy_B_half_inv,eTx_B_half_inv,eTy_B_half_inv)
    eT_A = cat([1,2],eTx_A,eTy_A,eTx_A,eTy_A)
    eT_B = cat([1,2],eTx_B,eTy_B,eTx_B,eTy_B)
    eT_A_inv = cat([1,2],eTx_A_inv,eTy_A_inv,eTx_A_inv,eTy_A_inv)
    eT_B_inv = cat([1,2],eTx_B_inv,eTy_B_inv,eTx_B_inv,eTy_B_inv)
  else
    eT_A_half = cat([1,2],eTx_A_half,eTy_A_half)
    eT_B_half = cat([1,2],eTx_B_half,eTy_B_half)
    eT_A_half_inv = cat([1,2],eTx_A_half_inv,eTy_A_half_inv)
    eT_B_half_inv = cat([1,2],eTx_B_half_inv,eTy_B_half_inv)
    eT_A = cat([1,2],eTx_A,eTy_A)
    eT_B = cat([1,2],eTx_B,eTy_B)
    eT_A_inv = cat([1,2],eTx_A_inv,eTy_A_inv)
    eT_B_inv = cat([1,2],eTx_B_inv,eTy_B_inv)
  end

  l.chkr_hop_half = [eT_A_half, eT_B_half]
  l.chkr_hop_half_inv = [eT_A_half_inv, eT_B_half_inv]
  l.chkr_hop_half_dagger = [ctranspose(eT_A_half), ctranspose(eT_B_half)]
  l.chkr_hop = [eT_A, eT_B]
  l.chkr_hop_inv = [eT_A_inv, eT_B_inv]
  l.chkr_hop_dagger = [ctranspose(eT_A), ctranspose(eT_B)]

  l.chkr_mu_half = spdiagm(fill(exp(-0.5*p.delta_tau * -p.mu), p.flv * l.sites))
  l.chkr_mu_half_inv = spdiagm(fill(exp(0.5*p.delta_tau * -p.mu), p.flv * l.sites))
  l.chkr_mu = spdiagm(fill(exp(-p.delta_tau * -p.mu), p.flv * l.sites))
  l.chkr_mu_inv = spdiagm(fill(exp(p.delta_tau * -p.mu), p.flv * l.sites))

  hop_mat_exp_chkr = l.chkr_hop_half[1] * l.chkr_hop_half[2] * sqrt.(l.chkr_mu)
  r = effreldiff(l.hopping_matrix_exp,hop_mat_exp_chkr)
  r[find(x->x==zero(x),hop_mat_exp_chkr)] = 0.
  println("Checkerboard - exact (abs):\t\t", maximum(absdiff(l.hopping_matrix_exp,hop_mat_exp_chkr)))
  # println("Checkerboard - exact (eff rel):\t\t", maximum(r))
end