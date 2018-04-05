# Beff(slice) = exp(−1/2∆τT)exp(−1/2∆τT)exp(−∆τV(slice))
function slice_matrix_no_chkr(mc::DQMC{C}, slice::Int, power::Float64=1.) where C<:CBFalse
  const eT = mc.l.hopping_matrix_exp
  const eTinv = mc.l.hopping_matrix_exp_inv
  const eV = interaction_matrix_exp(mc, slice, power)

  if power > 0
    return eT * eT * eV
  else
    return eV * eTinv * eTinv
  end
end