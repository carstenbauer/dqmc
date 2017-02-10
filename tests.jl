function test_calculate_boson_action_diff(p,l)
  old_hsfield = copy(p.hsfield)
  new_hsfield = copy(old_hsfield)

  site = rand(1:l.sites)
  slice = rand(1:p.slices)
  new_op = rand(3)
  # new_op = [100.,0.2,1234.56]
  new_hsfield[:,site,slice] = new_op[:]

  Sbef = calculate_boson_action(p,l,old_hsfield)
  Saft = calculate_boson_action(p,l,new_hsfield)
  dS_direct = Saft - Sbef
  dS = calculate_boson_action_diff(p,l,site,new_op,old_hsfield,slice)
  if dS==dS_direct
    error("Inconsistency between boson_action and boson_action_diff!")
  end
  return true
end
test_calculate_boson_action_diff(p,l)
# Worked


function test_interaction_matrix_consistency(p,l)
  for i in 1:l.sites
    if interaction_matrix_exp(p,l,s.current_slice)[i:l.sites:end,i:l.sites:end] != interaction_matrix_exp_op(p,l,p.hsfield[:,i,s.current_slice])
      error("Interaction_matrix Inconsistency!")
    end
  end
  return true
end
test_interaction_matrix_consistency(p,l)
# Worked


function delta_naive(s::Stack, p::Parameters, l::Lattice, i::Int, new_op::Vector{Float64})
  V1 = interaction_matrix_exp(p,l,s.current_slice,-1.0)
  bkp = copy(p.hsfield[:,i,s.current_slice])
  p.hsfield[:,i,s.current_slice] = new_op[:]
  V2 = interaction_matrix_exp(p,l,s.current_slice)
  p.hsfield[:,i,s.current_slice] = bkp[:]
  return (V1 * V2 - eye(p.flv*l.sites,p.flv*l.sites))
end


function delta_i_naive(s::Stack, p::Parameters, l::Lattice, i::Int, new_op::Vector{Float64})
  return delta_naive(s,p,l,i,new_op)[i:l.sites:end,i:l.sites:end]
end


function test_delta_i_consistency(s,p,l)
  old_hsfield = copy(p.hsfield)

  site = rand(1:l.sites)
  slice = s.current_slice
  # new_op = [1.,2.,3.]
  new_op = rand(3)

  detratio = calculate_detratio(s,p,l,site,new_op)
  delta_n = delta_naive(s,p,l,site,new_op)
  delta_i_n = delta_i_naive(s,p,l,site,new_op)

  if !isapprox(delta_i_n,s.delta_i)
    error("Efficient calculation of delta_i inconsistent with naive calculation!")
  elseif delta_n[site:l.sites:end,site:l.sites:end] != s.delta_i
    error("Delta_i inconsistent with full Delta!")
  end
  return true
end
test_delta_i_consistency(s,p,l)
# Worked
