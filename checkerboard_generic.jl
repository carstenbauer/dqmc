if !isdefined(:HoppingType)
  global const HoppingType = Complex128; # assume O(2) or O(3)
  warn("HoppingType wasn't set on loading checkerboard.jl")
  println("HoppingType = ", HoppingType)
end

# helper to cutoff numerical zeros
rem_eff_zeros!(X::AbstractArray) = map!(e->abs.(e)<1e-15?zero(e):e,X,X)

function init_checkerboard_matrices(p::Parameters, l::Lattice)
  println("Initializing hopping exponentials (Checkerboard, generic)")

  const n_groups = l.n_groups
  eT_half = Array{SparseMatrixCSC{HoppingType, Int}, 3}(n_groups,2,2) # group, spin (up, down), flavor (x, y)
  eT_half_inv = Array{SparseMatrixCSC{HoppingType, Int}, 3}(n_groups,2,2)
  eT = Array{SparseMatrixCSC{HoppingType, Int}, 3}(n_groups,2,2)
  eT_inv = Array{SparseMatrixCSC{HoppingType, Int}, 3}(n_groups,2,2)

  const VER = [0.0, 1.0]
  const HOR = [1.0, 0.0]

  for f in 1:2
    for s in 1:2
      for (g, gr) in enumerate(l.groups)
        # Build hopping matrix of individual chkr group
        T = zeros(HoppingType, l.sites, l.sites)

        for i in gr
          src = l.checkerboard[1,i]
          trg = l.checkerboard[2,i]
          bond = l.checkerboard[3,i]
          v = l.bond_vecs[bond,:]

          if v == VER
            T[trg, src] = T[src, trg] += -l.t[2,f]
          elseif v == HOR
            T[trg, src] = T[src, trg] += -l.t[1,f]
          else
            error("Square lattice??? Check lattice file!", v)
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

  l.chkr_mu_half = spdiagm(fill(exp(-0.5*p.delta_tau * -p.mu), p.flv * l.sites))
  l.chkr_mu_half_inv = spdiagm(fill(exp(0.5*p.delta_tau * -p.mu), p.flv * l.sites))
  l.chkr_mu = spdiagm(fill(exp(-p.delta_tau * -p.mu), p.flv * l.sites))
  l.chkr_mu_inv = spdiagm(fill(exp(p.delta_tau * -p.mu), p.flv * l.sites))

  hop_mat_exp_chkr = foldl(*,l.chkr_hop_half) * sqrt.(l.chkr_mu)
  r = effreldiff(l.hopping_matrix_exp,hop_mat_exp_chkr)
  r[find(x->x==zero(x),hop_mat_exp_chkr)] = 0.
  println("Checkerboard (generic) - exact (abs):\t\t", maximum(absdiff(l.hopping_matrix_exp,hop_mat_exp_chkr)))
end

#### WITH ARTIFICIAL B-FIELD

function init_checkerboard_matrices_Bfield(p::Parameters, l::Lattice)
  println("Initializing hopping exponentials (Bfield, Checkerboard, generic)")

  const n_groups = l.n_groups
  eT_half = Array{SparseMatrixCSC{HoppingType, Int}, 3}(n_groups,2,2) # group, spin (up, down), flavor (x, y)
  eT_half_inv = Array{SparseMatrixCSC{HoppingType, Int}, 3}(n_groups,2,2)
  eT = Array{SparseMatrixCSC{HoppingType, Int}, 3}(n_groups,2,2)
  eT_inv = Array{SparseMatrixCSC{HoppingType, Int}, 3}(n_groups,2,2)

  B = zeros(2,2) # rowidx = spin up,down, colidx = flavor
  if p.Bfield
    B[1,1] = B[2,2] = 2 * pi / l.sites
    B[1,2] = B[2,1] = - 2 * pi / l.sites
  end

  const VER = [0.0, 1.0]
  const HOR = [1.0, 0.0]

  for f in 1:2
    for s in 1:2
      for (g, gr) in enumerate(l.groups)
        # Build hopping matrix of individual chkr group
        T = zeros(HoppingType, l.sites, l.sites)

        for i in gr
          src = l.checkerboard[1,i]
          trg = l.checkerboard[2,i]
          bond = l.checkerboard[3,i]
          v = l.bond_vecs[bond,:]

          if v == VER
            T[trg, src] += - exp(im * l.peirls[s,f][trg,src]) * l.t[2,f]
            T[src, trg] += - exp(im * l.peirls[s,f][src,trg]) * l.t[2,f]
          elseif v == HOR
            T[trg, src] += - exp(im * l.peirls[s,f][trg,src]) * l.t[1,f]
            T[src, trg] += - exp(im * l.peirls[s,f][src,trg]) * l.t[1,f]
          else
            error("Square lattice??? Check lattice file!", v)
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

  l.chkr_mu_half = spdiagm(fill(exp(-0.5*p.delta_tau * -p.mu), p.flv * l.sites))
  l.chkr_mu_half_inv = spdiagm(fill(exp(0.5*p.delta_tau * -p.mu), p.flv * l.sites))
  l.chkr_mu = spdiagm(fill(exp(-p.delta_tau * -p.mu), p.flv * l.sites))
  l.chkr_mu_inv = spdiagm(fill(exp(p.delta_tau * -p.mu), p.flv * l.sites))

  hop_mat_exp_chkr = foldl(*,l.chkr_hop_half) * sqrt.(l.chkr_mu)
  r = effreldiff(l.hopping_matrix_exp,hop_mat_exp_chkr)
  r[find(x->x==zero(x),hop_mat_exp_chkr)] = 0.
  println("Checkerboard (Bfield, generic) - exact (abs):\t\t", maximum(absdiff(l.hopping_matrix_exp,hop_mat_exp_chkr)))
end

"""
Slice matrix
"""
function slice_matrix(s::Stack, p::Parameters, l::Lattice, slice::Int, power::Float64=1.)
  res = eye(HoppingType, p.flv*l.sites)
  if power > 0
    multiply_slice_matrix_left!(s,p, l, slice, res)
  else
    multiply_slice_matrix_inv_left!(s,p, l, slice, res)
  end
  return res
end


function multiply_slice_matrix_left!(s::Stack, p::Parameters, l::Lattice, slice::Int, M::AbstractMatrix{T}) where T<:Number

  interaction_matrix_exp!(s,p,l,slice,1.,s.eV)
  M[:] = s.eV * M
  M[:] = l.chkr_mu * M

  @inbounds @views begin
    for i in reverse(2:l.n_groups)
      M[:] = l.chkr_hop_half[i] * M
    end
    M[:] = l.chkr_hop[1] * M
    for i in 2:l.n_groups
      M[:] = l.chkr_hop_half[i] * M
    end
  end
end

function multiply_slice_matrix_right!(s::Stack, p::Parameters, l::Lattice, slice::Int, M::AbstractMatrix{T}) where T<:Number

  @inbounds @views begin
    for i in reverse(2:l.n_groups)
      M[:] = M * l.chkr_hop_half[i]
    end
    M[:] = M * l.chkr_hop[1]
    for i in 2:l.n_groups
      M[:] = M * l.chkr_hop_half[i]
    end
  end

  interaction_matrix_exp!(s,p,l,slice,1.,s.eV)
  M[:] = M * l.chkr_mu
  M[:] = M * s.eV
end

function multiply_slice_matrix_inv_left!(s::Stack, p::Parameters, l::Lattice, slice::Int, M::AbstractMatrix{T}) where T<:Number

  @inbounds @views begin
    for i in reverse(2:l.n_groups)
      M[:] = l.chkr_hop_half_inv[i] * M
    end
    M[:] = l.chkr_hop_inv[1] * M
    for i in 2:l.n_groups
      M[:] = l.chkr_hop_half_inv[i] * M
    end
  end

  interaction_matrix_exp!(s,p, l, slice, -1., s.eV)
  M[:] = l.chkr_mu_inv * M
  M[:] = s.eV * M
end

function multiply_slice_matrix_inv_right!(s::Stack, p::Parameters, l::Lattice, slice::Int, M::AbstractMatrix{T}) where T<:Number

  interaction_matrix_exp!(s,p, l, slice, -1., s.eV)
  M[:] = M * s.eV
  M[:] = M * l.chkr_mu_inv

  @inbounds @views begin
    for i in reverse(2:l.n_groups)
      M[:] = M * l.chkr_hop_half_inv[i]
    end
    M[:] = M * l.chkr_hop_inv[1]
    for i in 2:l.n_groups
      M[:] = M * l.chkr_hop_half_inv[i]
    end
  end
end

function multiply_daggered_slice_matrix_left!(s::Stack, p::Parameters, l::Lattice, slice::Int, M::AbstractMatrix{T}) where T<:Number

  
  @inbounds @views begin
    for i in reverse(2:l.n_groups)
      M[:] = l.chkr_hop_half_dagger[i] * M
    end
    M[:] = l.chkr_hop_dagger[1] * M
    for i in 2:l.n_groups
      M[:] = l.chkr_hop_half_dagger[i] * M
    end
  end

  interaction_matrix_exp!(s,p, l, slice, 1., s.eV)
  # s.eV == ctranspose(s.eV) and l.chkr_mu == ctranspose(s.chkr_mu)
  M[:] = l.chkr_mu * M
  M[:] = s.eV * M

end


function multiply_slice_matrix_left(s::Stack, p::Parameters, l::Lattice, slice::Int, M::AbstractMatrix{T}) where T<:Number
  X = copy(M)
  multiply_slice_matrix_left!(s,p, l, slice, X)
  return X
end

function multiply_slice_matrix_right(s::Stack, p::Parameters, l::Lattice, slice::Int, M::AbstractMatrix{T}) where T<:Number
  X = copy(M)
  multiply_slice_matrix_right!(s,p, l, slice, X)
  return X
end

function multiply_slice_matrix_inv_left(s::Stack, p::Parameters, l::Lattice, slice::Int, M::AbstractMatrix{T}) where T<:Number
  X = copy(M)
  multiply_slice_matrix_inv_left!(s,p, l, slice, X)
  return X
end

function multiply_slice_matrix_inv_right(s::Stack, p::Parameters, l::Lattice, slice::Int, M::AbstractMatrix{T}) where T<:Number
  X = copy(M)
  multiply_slice_matrix_inv_right!(s,p, l, slice, X)
  return X
end