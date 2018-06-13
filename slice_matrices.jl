# -------------------------------------------------------
#                        Generic
# -------------------------------------------------------
function slice_matrix(mc::AbstractDQMC, slice::Int, power::Float64=1.)
  const a = mc.a

  res = eye(heltype(mc), mc.p.flv*mc.l.sites)

  @mytimeit a.to "slice_matrix" begin
  if power > 0
    multiply_B_left!(mc, slice, res)
  else
    multiply_B_inv_left!(mc, slice, res)
  end
  end
  return res
end



# -------------------------------------------------------
#  				No checkerboard
# -------------------------------------------------------
# Beff(slice) = exp(−1/2∆τT)exp(−1/2∆τT)exp(−∆τV(slice))
function slice_matrix!(mc::DQMC_CBFalse, slice::Int, power::Float64=1., Bl::AbstractMatrix=mc.s.Bl)
  const eT = mc.l.hopping_matrix_exp
  const eTinv = mc.l.hopping_matrix_exp_inv
  const eV = mc.s.eV
  const tmp = mc.s.tmp

  interaction_matrix_exp!(mc, slice, power, eV)
  
  if power > 0
    A_mul_B!(tmp, eT, eV)
    A_mul_B!(Bl, eT, tmp)
    # return eT * eT * eV
  else
    A_mul_B!(tmp, eTinv, eTinv)
    A_mul_B!(Bl, eV, tmp)
    # return eV * eTinv * eTinv
  end
  nothing # slice_matrix now in Bl
end

function multiply_B_left!(mc::DQMC_CBFalse, slice::Int, M::AbstractMatrix)
  @mytimeit mc.a.to "multiply_B (combined)" begin
  @mytimeit mc.a.to "multiply_B_left!" begin
  slice_matrix!(mc, slice, 1., mc.s.Bl)
  A_mul_B!(mc.s.tmp, mc.s.Bl, M)
  M .= mc.s.tmp
  end #timeit
  end #timeit
  nothing
end
function multiply_B_right!(mc::DQMC_CBFalse, slice::Int, M::AbstractMatrix)
  @mytimeit mc.a.to "multiply_B (combined)" begin
  @mytimeit mc.a.to "multiply_B_right!" begin
  slice_matrix!(mc, slice, 1., mc.s.Bl)
  A_mul_B!(mc.s.tmp, M, mc.s.Bl)
  M .= mc.s.tmp
	end #timeit
  end #timeit
  nothing
end
function multiply_B_inv_right!(mc::DQMC_CBFalse, slice::Int, M::AbstractMatrix)
  @mytimeit mc.a.to "multiply_B (combined)" begin
  @mytimeit mc.a.to "multiply_B_inv_right!" begin
  slice_matrix!(mc, slice, -1., mc.s.Bl)
  A_mul_B!(mc.s.tmp, M, mc.s.Bl)
  M .= mc.s.tmp
	end #timeit
  end #timeit
  nothing
end
function multiply_B_inv_left!(mc::DQMC_CBFalse, slice::Int, M::AbstractMatrix)
  @mytimeit mc.a.to "multiply_B (combined)" begin
  @mytimeit mc.a.to "multiply_B_inv_left!" begin
  slice_matrix!(mc, slice, -1., mc.s.Bl)
  A_mul_B!(mc.s.tmp, mc.s.Bl, M)
  M .= mc.s.tmp
	end #timeit
  end #timeit
  nothing
end
function multiply_daggered_B_left!(mc::DQMC_CBFalse, slice::Int, M::AbstractMatrix)
  @mytimeit mc.a.to "multiply_B (combined)" begin
  @mytimeit mc.a.to "multiply_daggered_B_left!" begin
  slice_matrix!(mc, slice, 1., mc.s.Bl)
  Ac_mul_B!(mc.s.tmp, mc.s.Bl, M) # ctranspose
  M .= mc.s.tmp
  end #timeit
  end #timeit
  nothing
end



# -------------------------------------------------------
#  				Assaad checkerboard
# -------------------------------------------------------
function multiply_B_left!(mc::AbstractDQMC{CBAssaad}, slice::Int, M::AbstractMatrix{T}) where T<:Number
  const l = mc.l
  const s = mc.s
  const a = mc.a
  @mytimeit a.to "multiply_B (combined)" begin
  @mytimeit a.to "multiply_B_left!" begin

  interaction_matrix_exp!(mc,slice,1.,s.eV)
  A_mul_B!(s.tmp, s.eV, M)
  M .= s.tmp
  A_mul_B!(s.tmp, l.chkr_mu, M)
  M .= s.tmp

  A_mul_B!(s.tmp, l.chkr_hop_half[2], M)
  M .= s.tmp
  A_mul_B!(s.tmp, l.chkr_hop[1], M)
  M .= s.tmp
  A_mul_B!(s.tmp, l.chkr_hop_half[2], M)
  M .= s.tmp
  end
  end #timeit
  nothing
end

function multiply_B_right!(mc::AbstractDQMC{CBAssaad}, slice::Int, M::AbstractMatrix{T}) where T<:Number
  const l = mc.l
  const s = mc.s
  const a = mc.a
  @mytimeit a.to "multiply_B (combined)" begin
  @mytimeit a.to "multiply_B_right!" begin

  interaction_matrix_exp!(mc,slice,1.,s.eV)
  A_mul_B!(s.tmp, M, l.chkr_hop_half[2])
  M .= s.tmp
  A_mul_B!(s.tmp, M, l.chkr_hop[1])
  M .= s.tmp
  A_mul_B!(s.tmp, M, l.chkr_hop_half[2])
  M .= s.tmp

  A_mul_B!(s.tmp, M, l.chkr_mu)
  M .= s.tmp
  A_mul_B!(s.tmp, M, s.eV)
  M .= s.tmp
  end
  end #timeit
  nothing
end

function multiply_B_inv_left!(mc::AbstractDQMC{CBAssaad}, slice::Int, M::AbstractMatrix{T}) where T<:Number
  const l = mc.l
  const s = mc.s
  const a = mc.a
  @mytimeit a.to "multiply_B (combined)" begin
  @mytimeit a.to "multiply_B_inv_left!" begin
  
  interaction_matrix_exp!(mc, slice, -1., s.eV)
  A_mul_B!(s.tmp, l.chkr_hop_half_inv[2], M)
  M .= s.tmp
  A_mul_B!(s.tmp, l.chkr_hop_inv[1], M)
  M .= s.tmp
  A_mul_B!(s.tmp, l.chkr_hop_half_inv[2], M)
  M .= s.tmp

  A_mul_B!(s.tmp, l.chkr_mu_inv, M)
  M .= s.tmp
  A_mul_B!(s.tmp, s.eV, M)
  M .= s.tmp
  end
  end #timeit
  nothing
end

function multiply_B_inv_right!(mc::AbstractDQMC{CBAssaad}, slice::Int, M::AbstractMatrix{T}) where T<:Number
  const l = mc.l
  const s = mc.s
  const a = mc.a
  @mytimeit a.to "multiply_B (combined)" begin
  @mytimeit a.to "multiply_B_inv_right!" begin
  
  interaction_matrix_exp!(mc, slice, -1., s.eV)
  A_mul_B!(s.tmp, M, s.eV)
  M .= s.tmp
  A_mul_B!(s.tmp, M, l.chkr_mu_inv)
  M .= s.tmp

  A_mul_B!(s.tmp, M, l.chkr_hop_half_inv[2])
  M .= s.tmp
  A_mul_B!(s.tmp, M, l.chkr_hop_inv[1])
  M .= s.tmp
  A_mul_B!(s.tmp, M, l.chkr_hop_half_inv[2])
  M .= s.tmp
  end
  end #timeit
  nothing
end

function multiply_daggered_B_left!(mc::AbstractDQMC{CBAssaad}, slice::Int, M::AbstractMatrix{T}) where T<:Number
  const l = mc.l
  const s = mc.s
  const a = mc.a
  @mytimeit a.to "multiply_B (combined)" begin
  @mytimeit a.to "multiply_daggered_B_left!" begin
  
  interaction_matrix_exp!(mc, slice, 1., s.eV)
  A_mul_B!(s.tmp, l.chkr_hop_half_dagger[2], M)
  M .= s.tmp
  A_mul_B!(s.tmp, l.chkr_hop_dagger[1], M)
  M .= s.tmp
  A_mul_B!(s.tmp, l.chkr_hop_half_dagger[2], M)
  M .= s.tmp

  # s.eV == ctranspose(s.eV) and l.chkr_mu == ctranspose(s.chkr_mu)
  A_mul_B!(s.tmp, l.chkr_mu, M)
  M .= s.tmp
  A_mul_B!(s.tmp, s.eV, M)
  M .= s.tmp
  end
  end #timeit
  nothing
end



# -------------------------------------------------------
#  				Generic checkerboard
# -------------------------------------------------------
@inline function multiply_B_left!(mc::AbstractDQMC{CBGeneric}, slice::Int, M::AbstractMatrix{T}) where T<:Number
  const l = mc.l
  const p = mc.p
  const s = mc.s
  const a = mc.a
  @mytimeit a.to "multiply_B (combined)" begin
  @mytimeit a.to "multiply_B_left!" begin

  interaction_matrix_exp!(mc,slice,1.,s.eV)
  A_mul_B!(s.tmp, s.eV, M)
  M .= s.tmp
  A_mul_B!(s.tmp, l.chkr_mu, M)
  M .= s.tmp

  @inbounds @views begin
    for i in reverse(2:l.n_groups)
      A_mul_B!(s.tmp, l.chkr_hop_half[i], M)
      M .= s.tmp
    end
    A_mul_B!(s.tmp, l.chkr_hop[1], M)
    M .= s.tmp
    for i in 2:l.n_groups
      A_mul_B!(s.tmp, l.chkr_hop_half[i], M)
      M .= s.tmp
    end
  end
  end
  end #timeit
  nothing
end

function multiply_B_right!(mc::AbstractDQMC{CBGeneric}, slice::Int, M::AbstractMatrix{T}) where T<:Number
  const l = mc.l
  const p = mc.p
  const s = mc.s
  const a = mc.a
  @mytimeit a.to "multiply_B (combined)" begin
  @mytimeit a.to "multiply_B_right!" begin

  @inbounds @views begin
    for i in reverse(2:l.n_groups)
      A_mul_B!(s.tmp, M, l.chkr_hop_half[i])
      M .= s.tmp
    end
    A_mul_B!(s.tmp, M, l.chkr_hop[1])
    M .= s.tmp
    for i in 2:l.n_groups
      A_mul_B!(s.tmp, M, l.chkr_hop_half[i])
      M .= s.tmp
    end
  end

  interaction_matrix_exp!(mc,slice,1.,s.eV)
  A_mul_B!(s.tmp, M, l.chkr_mu)
  M .= s.tmp
  A_mul_B!(s.tmp, M, s.eV)
  M .= s.tmp
  end
  end #timeit
  nothing
end

function multiply_B_inv_left!(mc::AbstractDQMC{CBGeneric}, slice::Int, M::AbstractMatrix{T}) where T<:Number
  const l = mc.l
  const p = mc.p
  const s = mc.s
  const a = mc.a
  @mytimeit a.to "multiply_B (combined)" begin
  @mytimeit a.to "multiply_B_inv_left!" begin
  
  @inbounds @views begin
    for i in reverse(2:l.n_groups)
      A_mul_B!(s.tmp, l.chkr_hop_half_inv[i], M)
      M .= s.tmp
    end
    A_mul_B!(s.tmp, l.chkr_hop_inv[1], M)
    M .= s.tmp
    for i in 2:l.n_groups
      A_mul_B!(s.tmp, l.chkr_hop_half_inv[i], M)
      M .= s.tmp
    end
  end

  interaction_matrix_exp!(mc, slice, -1., s.eV)
  A_mul_B!(s.tmp, l.chkr_mu_inv, M)
  M .= s.tmp
  A_mul_B!(s.tmp, s.eV, M)
  M .= s.tmp
  end
  end #timeit
  nothing
end

function multiply_B_inv_right!(mc::AbstractDQMC{CBGeneric}, slice::Int, M::AbstractMatrix{T}) where T<:Number
  const l = mc.l
  const p = mc.p
  const s = mc.s
  const a = mc.a
  @mytimeit a.to "multiply_B (combined)" begin
  @mytimeit a.to "multiply_B_inv_right!" begin
  
  interaction_matrix_exp!(mc, slice, -1., s.eV)
  A_mul_B!(s.tmp, M, s.eV)
  M .= s.tmp
  A_mul_B!(s.tmp, M, l.chkr_mu_inv)
  M .= s.tmp

  @inbounds @views begin
    for i in reverse(2:l.n_groups)
      A_mul_B!(s.tmp, M, l.chkr_hop_half_inv[i])
      M .= s.tmp
    end
    A_mul_B!(s.tmp, M, l.chkr_hop_inv[1])
    M .= s.tmp
    for i in 2:l.n_groups
      A_mul_B!(s.tmp, M, l.chkr_hop_half_inv[i])
      M .= s.tmp
    end
  end
  end
  end #timeit
  nothing
end

function multiply_daggered_B_left!(mc::AbstractDQMC{CBGeneric}, slice::Int, M::AbstractMatrix{T}) where T<:Number
  const l = mc.l
  const p = mc.p
  const s = mc.s
  const a = mc.a
  @mytimeit a.to "multiply_B (combined)" begin
  @mytimeit a.to "multiply_daggered_B_left!" begin
    
  @inbounds @views begin
    for i in reverse(2:l.n_groups)
      A_mul_B!(s.tmp, l.chkr_hop_half_dagger[i], M)
      M .= s.tmp
    end
    A_mul_B!(s.tmp, l.chkr_hop_dagger[1], M)
    M .= s.tmp
    for i in 2:l.n_groups
      A_mul_B!(s.tmp, l.chkr_hop_half_dagger[i], M)
      M .= s.tmp
    end
  end

  interaction_matrix_exp!(mc, slice, 1., s.eV)
  # s.eV == ctranspose(s.eV) and l.chkr_mu == ctranspose(s.chkr_mu)
  A_mul_B!(s.tmp, l.chkr_mu, M)
  M .= s.tmp
  A_mul_B!(s.tmp, s.eV, M)
  M .= s.tmp
  end
  end #timeit
  nothing
end