function build_checkerboard(mc::AbstractDQMC{CBGeneric})
  l = mc.l

  l.groups = UnitRange[]
  edges_used = zeros(Int64, l.n_bonds)
  l.checkerboard = zeros(3, l.n_bonds)
  group_start = 1
  group_end = 1

  while minimum(edges_used) == 0
    sites_used = zeros(Int64, l.sites)

    for id in 1:l.n_bonds
      src, trg, typ = l.bonds[id,1:3]

      if edges_used[id] == 1 continue end
      if sites_used[src] == 1 continue end
      if sites_used[trg] == 1 continue end

      edges_used[id] = 1
      sites_used[src] = 1
      sites_used[trg] = 1

      l.checkerboard[:, group_end] = [src, trg, id]
      group_end += 1
    end
    push!(l.groups, group_start:group_end-1)
    group_start = group_end
  end
  l.n_groups = length(l.groups)
end


# helper to cutoff numerical zeros
rem_eff_zeros!(X::AbstractArray) = map!(e -> abs.(e) < 1e-15 ? zero(e) : e,X,X)

function init_checkerboard_matrices(mc::AbstractDQMC{CBGeneric})
  l = mc.l
  p = mc.p
  H = heltype(mc)

  println("Initializing hopping exponentials (Checkerboard, generic)")

  build_checkerboard(mc)

  n_groups = l.n_groups
  eT_half = Array{SparseMatrixCSC{H, Int}, 3}(n_groups,2,2) # group, spin (up, down), flavor (x, y)
  eT_half_inv = Array{SparseMatrixCSC{H, Int}, 3}(n_groups,2,2)
  eT = Array{SparseMatrixCSC{H, Int}, 3}(n_groups,2,2)
  eT_inv = Array{SparseMatrixCSC{H, Int}, 3}(n_groups,2,2)

  U = [0.0, 1.0]
  R = [1.0, 0.0]
  UR = [1.0, 1.0]
  DR = [1.0, -1.0]
  UU = [0.0, 2.0]
  RR = [2.0, 0.0]

  for f in 1:2
    for s in 1:2
      for (g, gr) in enumerate(l.groups)
        # Build hopping matrix of individual chkr group
        T = zeros(H, l.sites, l.sites)

        for i in gr
          src = l.checkerboard[1,i]
          trg = l.checkerboard[2,i]
          bond = l.checkerboard[3,i]
          v = l.bond_vecs[bond,:]

          # nn
          if v == U && p.hoppings != "none"
            T[trg, src] = T[src, trg] += -l.t[2,f]
          elseif v == R
            T[trg, src] = T[src, trg] += -l.t[1,f]

          #Nnn
          elseif (v == UR || v == DR) && p.Nhoppings != "none"
            T[trg, src] = T[src, trg] += -l.tN[1,f]

          #NNnn
          elseif v == UU && p.NNhoppings != "none"
            T[trg, src] = T[src, trg] += -l.tNN[2,f]
          elseif v == RR
            T[trg, src] = T[src, trg] += -l.tNN[1,f]
          end
        end

        eT_half[g,s,f] = sparse(rem_eff_zeros!(expm(- 0.5 * p.delta_tau * T)))
        eT_half_inv[g,s,f] = sparse(rem_eff_zeros!(expm(0.5 * p.delta_tau * T)))
        eT[g,s,f] = sparse(rem_eff_zeros!(expm(- p.delta_tau * T)))
        eT_inv[g,s,f] = sparse(rem_eff_zeros!(expm(p.delta_tau * T)))
      end
    end
  end

  if p.opdim == 3
    l.chkr_hop_half = [cat([1,2], eT_half[g,1,1], eT_half[g,2,2], eT_half[g,2,1], eT_half[g,1,2]) for g in 1:n_groups]
    l.chkr_hop_half_inv = [cat([1,2], eT_half_inv[g,1,1], eT_half_inv[g,2,2], eT_half_inv[g,2,1], eT_half_inv[g,1,2]) for g in 1:n_groups]
    l.chkr_hop = [cat([1,2], eT[g,1,1], eT[g,2,2], eT[g,2,1], eT[g,1,2]) for g in 1:n_groups]
    l.chkr_hop_inv = [cat([1,2], eT_inv[g,1,1], eT_inv[g,2,2], eT_inv[g,2,1], eT_inv[g,1,2]) for g in 1:n_groups]

  else # O(2) and O(1) model
    l.chkr_hop_half = [cat([1,2], eT_half[g,1,1], eT_half[g,2,2]) for g in 1:n_groups]
    l.chkr_hop_half_inv = [cat([1,2], eT_half_inv[g,1,1], eT_half_inv[g,2,2]) for g in 1:n_groups]
    l.chkr_hop = [cat([1,2], eT[g,1,1], eT[g,2,2]) for g in 1:n_groups]
    l.chkr_hop_inv = [cat([1,2], eT_inv[g,1,1], eT_inv[g,2,2]) for g in 1:n_groups]
  end
  l.chkr_hop_half_dagger = ctranspose.(l.chkr_hop_half)
  l.chkr_hop_dagger = ctranspose.(l.chkr_hop)

  muv = vcat(fill(p.mu1,l.sites),fill(p.mu2,l.sites))
  muv = repeat(muv, outer=[Int(p.flv/2)])
  l.chkr_mu_half = spdiagm(exp.(-0.5*p.delta_tau * -muv))
  l.chkr_mu_half_inv = spdiagm(exp.(0.5*p.delta_tau * -muv))
  l.chkr_mu = spdiagm(exp.(-p.delta_tau * -muv))
  l.chkr_mu_inv = spdiagm(exp.(p.delta_tau * -muv))

  fold_chkr_grps(mc)

  hop_mat_exp_chkr = foldl(*,l.chkr_hop_half) * sqrt.(l.chkr_mu)
  r = effreldiff(l.hopping_matrix_exp,hop_mat_exp_chkr)
  r[find(x->x==zero(x),hop_mat_exp_chkr)] = 0.
  println("Checkerboard (generic) - exact (abs):\t\t", maximum(absdiff(l.hopping_matrix_exp,hop_mat_exp_chkr)))
end



function fold_chkr_grps(mc::AbstractDQMC{CBGeneric})
  # ### Heuristic folding (merging) of chkr groups.
  #
  # We often have many sparse chkr groups which stay sparse, when folded (multiplied together).
  # It is advantageous to reduce the number of sparse matrices that we multiply all the time in
  # multiply_B_*! functions.
  #
  # The applied heuristic is as follows:
  # Fold chkr matrices until a SPARSITY_LIMIT is reached. Save the result as one "folded" matrix.
  # Continue in the same spirit with remaining chkr matrices until we visited them all.
  #
  # Massive(!) speed improvement!!
  l = mc.l
  H = heltype(mc)
  flv = mc.p.flv
  N = mc.l.sites
  SPARSITY_LIMIT = mc.p.sparsity_limit

  @show SPARSITY_LIMIT

  l.chkr_hop_half_folded = SparseMatrixCSC{H, Int64}[]
  cur = speye(H, flv*N, flv*N)
  l.folded = UnitRange{Int64}[]
  fstart = 2
  fstop = 1

  for i in 2:l.n_groups
    if i != 2 && (countnz(cur)/length(cur)) > SPARSITY_LIMIT
      push!(l.chkr_hop_half_folded, cur)
      push!(l.folded, fstart:fstop)
      fstart = i
      fstop = i
      cur = speye(H, flv*N, flv*N)
    else
      fstop += 1
    end

    cur = l.chkr_hop_half[i] * cur
  end

  push!(l.chkr_hop_half_folded, cur)
  push!(l.folded, fstart:fstop)

  l.n_folded = length(l.folded)

  l.chkr_hop_half_inv_folded = Vector{SparseMatrixCSC{H, Int64}}(l.n_folded)
  l.chkr_hop_half_dagger_folded = Vector{SparseMatrixCSC{H, Int64}}(l.n_folded)
  l.chkr_hop_half_folded_rev = Vector{SparseMatrixCSC{H, Int64}}(l.n_folded)
  l.chkr_hop_half_inv_folded_rev = Vector{SparseMatrixCSC{H, Int64}}(l.n_folded)
  l.chkr_hop_half_dagger_folded_rev = Vector{SparseMatrixCSC{H, Int64}}(l.n_folded)

  for (i, rng) in enumerate(l.folded)
    cur_rev = speye(H, flv*N, flv*N)
    cur_inv = speye(H, flv*N, flv*N)
    cur_inv_rev = speye(H, flv*N, flv*N)
    cur_dagger = speye(H, flv*N, flv*N)
    cur_dagger_rev = speye(H, flv*N, flv*N)
    for k in rng
      cur_rev = cur_rev * l.chkr_hop_half[k]
      cur_inv = l.chkr_hop_half_inv[k] * cur_inv
      cur_inv_rev = cur_inv_rev * l.chkr_hop_half_inv[k]
      cur_dagger = l.chkr_hop_half_dagger[k] * cur_dagger
      cur_dagger_rev = cur_dagger_rev * l.chkr_hop_half_dagger[k]
    end
    l.chkr_hop_half_folded_rev[i] = cur_rev
    l.chkr_hop_half_inv_folded[i] = cur_inv
    l.chkr_hop_half_inv_folded_rev[i] = cur_inv_rev
    l.chkr_hop_half_dagger_folded[i] = cur_dagger
    l.chkr_hop_half_dagger_folded_rev[i] = cur_dagger_rev
  end

  @show l.n_groups
  @show l.n_folded
  nothing
end

#### WITH ARTIFICIAL B-FIELD

function init_checkerboard_matrices_Bfield(mc::AbstractDQMC{CBGeneric})
  l = mc.l
  p = mc.p
  H = heltype(mc)

  println("Initializing hopping exponentials (Bfield, Checkerboard, generic)")

  build_checkerboard(mc)

  n_groups = l.n_groups
  eT_half = Array{SparseMatrixCSC{H, Int}, 3}(n_groups,2,2) # group, spin (up, down), flavor (x, y)
  eT_half_inv = Array{SparseMatrixCSC{H, Int}, 3}(n_groups,2,2)
  eT = Array{SparseMatrixCSC{H, Int}, 3}(n_groups,2,2)
  eT_inv = Array{SparseMatrixCSC{H, Int}, 3}(n_groups,2,2)

  B = zeros(2,2) # rowidx = spin up,down, colidx = flavor
  if p.Bfield
    B[1,1] = B[2,2] = 2 * pi / l.sites
    B[1,2] = B[2,1] = - 2 * pi / l.sites
  end

  U = [0.0, 1.0]
  R = [1.0, 0.0]
  UR = [1.0, 1.0]
  DR = [1.0, -1.0]
  UU = [0.0, 2.0]
  RR = [2.0, 0.0]

  for f in 1:2
    for s in 1:2
      for (g, gr) in enumerate(l.groups)
        # Build hopping matrix of individual chkr group
        T = zeros(H, l.sites, l.sites)

        for i in gr
          src = l.checkerboard[1,i]
          trg = l.checkerboard[2,i]
          bond = l.checkerboard[3,i]
          v = l.bond_vecs[bond,:]

          if v == U && p.hoppings != "none"
            T[trg, src] += - exp(im * l.peirls[s,f][trg,src]) * l.t[2,f]
            T[src, trg] += - exp(im * l.peirls[s,f][src,trg]) * l.t[2,f]
          elseif v == R && p.hoppings != "none"
            T[trg, src] += - exp(im * l.peirls[s,f][trg,src]) * l.t[1,f]
            T[src, trg] += - exp(im * l.peirls[s,f][src,trg]) * l.t[1,f]

          #Nnn
          elseif (v == UR || v == DR) && p.Nhoppings != "none"
            T[trg, src] += - exp(im * l.peirls[s,f][trg,src]) * l.tN[1,f]
            T[src, trg] += - exp(im * l.peirls[s,f][src,trg]) * l.tN[1,f]

          #NNnn
          elseif v == UU  && p.NNhoppings != "none"
            T[trg, src] += - exp(im * l.peirls[s,f][trg,src]) * l.tNN[2,f]
            T[src, trg] += - exp(im * l.peirls[s,f][src,trg]) * l.tNN[2,f]
          elseif v == RR && p.NNhoppings != "none"
            T[trg, src] += - exp(im * l.peirls[s,f][trg,src]) * l.tNN[1,f]
            T[src, trg] += - exp(im * l.peirls[s,f][src,trg]) * l.tNN[1,f]
          end
        end

        eT_half[g,s,f] = sparse(rem_eff_zeros!(expm(- 0.5 * p.delta_tau * T)))
        eT_half_inv[g,s,f] = sparse(rem_eff_zeros!(expm(0.5 * p.delta_tau * T)))
        eT[g,s,f] = sparse(rem_eff_zeros!(expm(- p.delta_tau * T)))
        eT_inv[g,s,f] = sparse(rem_eff_zeros!(expm(p.delta_tau * T)))
      end
    end
  end

  if p.opdim == 3
    l.chkr_hop_half = [cat([1,2], eT_half[g,1,1], eT_half[g,2,2], eT_half[g,2,1], eT_half[g,1,2]) for g in 1:n_groups]
    l.chkr_hop_half_inv = [cat([1,2], eT_half_inv[g,1,1], eT_half_inv[g,2,2], eT_half_inv[g,2,1], eT_half_inv[g,1,2]) for g in 1:n_groups]
    l.chkr_hop = [cat([1,2], eT[g,1,1], eT[g,2,2], eT[g,2,1], eT[g,1,2]) for g in 1:n_groups]
    l.chkr_hop_inv = [cat([1,2], eT_inv[g,1,1], eT_inv[g,2,2], eT_inv[g,2,1], eT_inv[g,1,2]) for g in 1:n_groups]

  else # O(2) and O(1) model
    l.chkr_hop_half = [cat([1,2], eT_half[g,1,1], eT_half[g,2,2]) for g in 1:n_groups]
    l.chkr_hop_half_inv = [cat([1,2], eT_half_inv[g,1,1], eT_half_inv[g,2,2]) for g in 1:n_groups]
    l.chkr_hop = [cat([1,2], eT[g,1,1], eT[g,2,2]) for g in 1:n_groups]
    l.chkr_hop_inv = [cat([1,2], eT_inv[g,1,1], eT_inv[g,2,2]) for g in 1:n_groups]
  end
  l.chkr_hop_half_dagger = ctranspose.(l.chkr_hop_half)
  l.chkr_hop_dagger = ctranspose.(l.chkr_hop)

  muv = vcat(fill(p.mu1,l.sites),fill(p.mu2,l.sites))
  muv = repeat(muv, outer=[Int(p.flv/2)])
  l.chkr_mu_half = spdiagm(exp.(-0.5*p.delta_tau * -muv))
  l.chkr_mu_half_inv = spdiagm(exp.(0.5*p.delta_tau * -muv))
  l.chkr_mu = spdiagm(exp.(-p.delta_tau * -muv))
  l.chkr_mu_inv = spdiagm(exp.(p.delta_tau * -muv))

  fold_chkr_grps(mc)

  hop_mat_exp_chkr = foldl(*,l.chkr_hop_half) * sqrt.(l.chkr_mu)
  r = effreldiff(l.hopping_matrix_exp,hop_mat_exp_chkr)
  r[find(x->x==zero(x),hop_mat_exp_chkr)] = 0.
  println("Checkerboard (Bfield, generic) - exact (abs):\t\t", maximum(absdiff(l.hopping_matrix_exp,hop_mat_exp_chkr)))
end