function global_update_backup_swap!(s::Stack, p::Parameters, l::Lattice)
  # swap current stack and greens to backup stack and greens (more efficient than copying)
  s.gb_u_stack, s.u_stack = s.u_stack, s.gb_u_stack
  s.gb_d_stack, s.d_stack = s.d_stack, s.gb_d_stack
  s.gb_vt_stack, s.vt_stack = s.vt_stack, s.gb_vt_stack
  s.gb_greens, s.greens = s.greens, s.gb_greens # this is greens at time slice == p.slices
  s.gb_greens_svs, s.greens_svs = s.greens_svs, s.gb_greens_svs # these are svs of greens at time slice == p.slices + 1
end


function global_update_perform_shift!(s::Stack, p::Parameters, l::Lattice)
  global_op_shift = rand(p.box, 3)
  for i in 1:l.sites
    for n in 1:p.slices
      p.hsfield[:,i,n] += global_op_shift
    end
  end
end


function global_update(s::Stack, p::Parameters, l::Lattice)

  S_old = calculate_boson_action(p, l)
  if S_old != p.boson_action warn("Incorrect boson action found during attempt to do global update.") end

  global_update_backup_swap!(s,p,l)
  s.gb_hsfield = copy(p.hsfield)

  global_update_perform_shift!(s,p,l) # TODO: Comment out and see if s.gb_greens is equl to s.greens after stack rebuild.
  update_interaction_sinh_cosh_all(p,l)
  # rebuild stack
  build_stack(s, p, l)
  # initial propagate
  propagate(s, p, l)
  # now we have s.greens = G_{p.slices} (up to one down-wrap) and s.greens_svs = svs of G_{p.slices + 1}

  p.boson_action = calculate_boson_action(p, l)
  p_boson = exp(-(p.boson_action - S_old)) # exp_delta_S_boson

  # calculate detratio = fermion accept. prob.
  log_prob = 0.
  for j in 1:p.flv*l.sites
      # TODO: max has different/wrong order here?
      log_prob += log(s.greens_svs[j]) - log(s.gb_greens_svs[j])
  end
  p_fermion = exp(log_prob)

  p_acc = p_boson * real(p_fermion)

  if p_acc > 1.0 || rand() < p_acc
    return true
  else
    p.boson_action = S_old
    p.hsfield = s.gb_hsfield
    global_update_backup_swap!(s,p,l)
    return false
  end

end