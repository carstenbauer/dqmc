# -------------------------------------------------------
#  				No checkerboard
# -------------------------------------------------------
# Beff(slice) = exp(−1/2∆τT)exp(−1/2∆τT)exp(−∆τV(slice))
function slice_matrix(mc::DQMC_CBFalse, slice::Int, power::Float64=1.)
  const eT = mc.l.hopping_matrix_exp
  const eTinv = mc.l.hopping_matrix_exp_inv
  const eV = interaction_matrix_exp(mc, slice, power)

  if power > 0
    return eT * eT * eV
  else
    return eV * eTinv * eTinv
  end
end

function multiply_slice_matrix_left!(mc::DQMC_CBFalse, slice::Int, M::AbstractMatrix)
	M .= slice_matrix(mc, slice, 1.) * M
	nothing
end
function multiply_slice_matrix_right!(mc::DQMC_CBFalse, slice::Int, M::AbstractMatrix)
	M .= M * slice_matrix(mc, slice, 1.)
	nothing
end
function multiply_slice_matrix_inv_right!(mc::DQMC_CBFalse, slice::Int, M::AbstractMatrix)
	M .= M * slice_matrix(mc, slice, -1.)
	nothing
end
function multiply_slice_matrix_inv_left!(mc::DQMC_CBFalse, slice::Int, M::AbstractMatrix)
	M .= slice_matrix(mc, slice, -1.) * M
	nothing
end
function multiply_daggered_slice_matrix_left!(mc::DQMC_CBFalse, slice::Int, M::AbstractMatrix)
	M .= ctranspose(slice_matrix(mc, slice, 1.)) * M
	nothing
end



# -------------------------------------------------------
#  				For all checkerboard variants
# -------------------------------------------------------
function slice_matrix(mc::DQMC_CBTrue, slice::Int, power::Float64=1.)
  res = eye(heltype(mc), mc.p.flv*mc.l.sites)
  if power > 0
    multiply_slice_matrix_left!(mc, slice, res)
  else
    multiply_slice_matrix_inv_left!(mc, slice, res)
  end
  return res
end

function multiply_slice_matrix_left(mc::DQMC_CBTrue, slice::Int, M::AbstractMatrix{T}) where T<:Number
  X = copy(M)
  multiply_slice_matrix_left!(mc, slice, X)
  return X
end

function multiply_slice_matrix_right(mc::DQMC_CBTrue, slice::Int, M::AbstractMatrix{T}) where T<:Number
  X = copy(M)
  multiply_slice_matrix_right!(mc, slice, X)
  return X
end

function multiply_slice_matrix_inv_left(mc::DQMC_CBTrue, slice::Int, M::AbstractMatrix{T}) where T<:Number
  X = copy(M)
  multiply_slice_matrix_inv_left!(mc, slice, X)
  return X
end

function multiply_slice_matrix_inv_right(mc::DQMC_CBTrue, slice::Int, M::AbstractMatrix{T}) where T<:Number
  X = copy(M)
  multiply_slice_matrix_inv_right!(mc, slice, X)
  return X
end



# -------------------------------------------------------
#  				Assaad checkerboard
# -------------------------------------------------------
function multiply_slice_matrix_left!(mc::AbstractDQMC{CBAssaad}, slice::Int, M::AbstractMatrix{T}) where T<:Number
  const l = mc.l
  const s = mc.s

  interaction_matrix_exp!(mc,slice,1.,s.eV)
  M[:] = s.eV * M
  M[:] = l.chkr_mu * M

  M[:] = l.chkr_hop_half[2] * M
  M[:] = l.chkr_hop[1] * M
  M[:] = l.chkr_hop_half[2] * M
  nothing
end

function multiply_slice_matrix_right!(mc::AbstractDQMC{CBAssaad}, slice::Int, M::AbstractMatrix{T}) where T<:Number
  const l = mc.l
  const s = mc.s

  interaction_matrix_exp!(mc,slice,1.,s.eV)
  M[:] = M * l.chkr_hop_half[2]
  M[:] = M * l.chkr_hop[1]
  M[:] = M * l.chkr_hop_half[2]

  M[:] = M * l.chkr_mu
  M[:] = M * s.eV
  nothing
end

function multiply_slice_matrix_inv_left!(mc::AbstractDQMC{CBAssaad}, slice::Int, M::AbstractMatrix{T}) where T<:Number
  const l = mc.l
  const s = mc.s
  
  interaction_matrix_exp!(mc, slice, -1., s.eV)
  M[:] = l.chkr_hop_half_inv[2] * M
  M[:] = l.chkr_hop_inv[1] * M
  M[:] = l.chkr_hop_half_inv[2] * M

  M[:] = l.chkr_mu_inv * M
  M[:] = s.eV * M
  nothing
end

function multiply_slice_matrix_inv_right!(mc::AbstractDQMC{CBAssaad}, slice::Int, M::AbstractMatrix{T}) where T<:Number
  const l = mc.l
  const s = mc.s
  
  interaction_matrix_exp!(mc, slice, -1., s.eV)
  M[:] = M * s.eV
  M[:] = M * l.chkr_mu_inv

  M[:] = M * l.chkr_hop_half_inv[2]
  M[:] = M * l.chkr_hop_inv[1]
  M[:] = M * l.chkr_hop_half_inv[2]
  nothing
end

function multiply_daggered_slice_matrix_left!(mc::AbstractDQMC{CBAssaad}, slice::Int, M::AbstractMatrix{T}) where T<:Number
  const l = mc.l
  const s = mc.s
  
  interaction_matrix_exp!(mc, slice, 1., s.eV)
  M[:] = l.chkr_hop_half_dagger[2] * M
  M[:] = l.chkr_hop_dagger[1] * M
  M[:] = l.chkr_hop_half_dagger[2] * M

  # s.eV == ctranspose(s.eV) and l.chkr_mu == ctranspose(s.chkr_mu)
  M[:] = l.chkr_mu * M
  M[:] = s.eV * M
  nothing
end



# -------------------------------------------------------
#  				Generic checkerboard
# -------------------------------------------------------
function multiply_slice_matrix_left!(mc::AbstractDQMC{CBGeneric}, slice::Int, M::AbstractMatrix{T}) where T<:Number
  const l = mc.l
  const p = mc.p
  const s = mc.s

  interaction_matrix_exp!(mc,slice,1.,s.eV)
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
  nothing
end

function multiply_slice_matrix_right!(mc::AbstractDQMC{CBGeneric}, slice::Int, M::AbstractMatrix{T}) where T<:Number
  const l = mc.l
  const p = mc.p
  const s = mc.s

  @inbounds @views begin
    for i in reverse(2:l.n_groups)
      M[:] = M * l.chkr_hop_half[i]
    end
    M[:] = M * l.chkr_hop[1]
    for i in 2:l.n_groups
      M[:] = M * l.chkr_hop_half[i]
    end
  end

  interaction_matrix_exp!(mc,slice,1.,s.eV)
  M[:] = M * l.chkr_mu
  M[:] = M * s.eV
  nothing
end

function multiply_slice_matrix_inv_left!(mc::AbstractDQMC{CBGeneric}, slice::Int, M::AbstractMatrix{T}) where T<:Number
  const l = mc.l
  const p = mc.p
  const s = mc.s
  
  @inbounds @views begin
    for i in reverse(2:l.n_groups)
      M[:] = l.chkr_hop_half_inv[i] * M
    end
    M[:] = l.chkr_hop_inv[1] * M
    for i in 2:l.n_groups
      M[:] = l.chkr_hop_half_inv[i] * M
    end
  end

  interaction_matrix_exp!(mc, slice, -1., s.eV)
  M[:] = l.chkr_mu_inv * M
  M[:] = s.eV * M
  nothing
end

function multiply_slice_matrix_inv_right!(mc::AbstractDQMC{CBGeneric}, slice::Int, M::AbstractMatrix{T}) where T<:Number
  const l = mc.l
  const p = mc.p
  const s = mc.s
  
  interaction_matrix_exp!(mc, slice, -1., s.eV)
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
  nothing
end

function multiply_daggered_slice_matrix_left!(mc::AbstractDQMC{CBGeneric}, slice::Int, M::AbstractMatrix{T}) where T<:Number
  const l = mc.l
  const p = mc.p
  const s = mc.s
    
  @inbounds @views begin
    for i in reverse(2:l.n_groups)
      M[:] = l.chkr_hop_half_dagger[i] * M
    end
    M[:] = l.chkr_hop_dagger[1] * M
    for i in 2:l.n_groups
      M[:] = l.chkr_hop_half_dagger[i] * M
    end
  end

  interaction_matrix_exp!(mc, slice, 1., s.eV)
  # s.eV == ctranspose(s.eV) and l.chkr_mu == ctranspose(s.chkr_mu)
  M[:] = l.chkr_mu * M
  M[:] = s.eV * M
  nothing
end

# TODO: resume