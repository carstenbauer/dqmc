function test_interaction_matrix_consistency(p,l)
  for i in 1:l.sites
    if interaction_matrix_exp(p,l,s.current_slice)[i:l.sites:end,i:l.sites:end] != interaction_matrix_exp_op(p,l,p.hsfield[:,i,s.current_slice])
      error("Interaction_matrix Inconsistency!")
    end
  end
  return true
end
# test_interaction_matrix_consistency(p,l)
# Worked
