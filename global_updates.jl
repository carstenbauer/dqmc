function global_update_backup_swap!(s::Stack, p::Parameters, l::Lattice)
  # swap current stack and greens to backup stack and greens (more efficient than copying)
  s.gb_u_stack, s.u_stack = s.u_stack, s.gb_u_stack
  s.gb_d_stack, s.d_stack = s.d_stack, s.gb_d_stack
  s.gb_vt_stack, s.vt_stack = s.vt_stack, s.gb_vt_stack
  s.gb_greens, s.greens = s.greens, s.gb_greens # this is greens at time slice == p.slices
  s.gb_greens_inv_svs, s.greens_inv_svs = s.greens_inv_svs, s.gb_greens_inv_svs # these are svs of greens at time slice == p.slices + 1
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

  assert(s.current_slice == p.slices && s.direction == -1)

  S_old = p.boson_action
  if !isapprox(S_old,calculate_boson_action(p, l)) warn("Incorrect boson action found during attempt to do global update.") end

  global_update_backup_swap!(s,p,l)
  s.gb_hsfield = copy(p.hsfield)

  global_update_perform_shift!(s,p,l)
  update_interaction_sinh_cosh_all(p,l)
  build_stack(s, p, l)
  propagate(s, p, l)
  # now we have s.greens = G_{p.slices} (up to one down-wrap) and s.greens_inv_svs = svs of G_{p.slices + 1}

  p.boson_action = calculate_boson_action(p, l)

  # @printf("S_new: %.2e\n", p.boson_action)
  # @printf("S_old: %.2e\n", S_old)
  # @printf("S_new - S_old: %.2e\n", p.boson_action - S_old)
  p_boson = exp(-(p.boson_action - S_old)) # exp_delta_S_boson

  # calculate detratio = fermion accept. prob.
  log_prob = 0.
  for j in 1:p.flv*l.sites
      log_prob += log(s.greens_inv_svs[j]) - log(s.gb_greens_inv_svs[j])
  end
  p_fermion = exp(log_prob)

  p_acc = p_boson * real(p_fermion)

  # @printf("p_boson %.2e\n",abs(p_boson))
  # @printf("p_fermion %.2e\n",abs(p_fermion))
  # @printf("p_acc %.2e\n",abs(p_acc))
  # println("")

  if p_acc > 1.0 || rand() < p_acc
    return 1
  else
    # undo global move
    p.boson_action = S_old
    p.hsfield = s.gb_hsfield
    global_update_backup_swap!(s,p,l)
    return 0
  end

end