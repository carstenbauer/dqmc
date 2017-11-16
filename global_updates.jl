function global_update_backup_swap!(s::Stack, p::Parameters, l::Lattice)
  # swap current stack and greens to backup stack and greens (more efficient than copying)
  s.gb_u_stack, s.u_stack = s.u_stack, s.gb_u_stack
  s.gb_d_stack, s.d_stack = s.d_stack, s.gb_d_stack
  s.gb_t_stack, s.t_stack = s.t_stack, s.gb_t_stack
  s.gb_greens, s.greens = s.greens, s.gb_greens # this is greens at time slice == p.slices
  s.gb_log_det, s.log_det = s.log_det, s.gb_log_det # this is logdet of greens at time slice == p.slices + 1
end

@inline function global_update_perform_shift!(s::Stack, p::Parameters, l::Lattice)
    @inbounds @views begin
      @simd for k in 1:p.opdim
        p.hsfield[k,:,:] .+= rand(p.box_global)
      end
    end
end

function global_update(s::Stack, p::Parameters, l::Lattice)

  @assert (s.current_slice == p.slices) && (s.direction == -1)

  S_old = p.boson_action
  # if p.all_checks && !isapprox(S_old,calculate_boson_action(p, l)) warn("Incorrect boson action found during attempt to do global update.") end

  global_update_backup_swap!(s,p,l) # save current stack etc.
  s.gb_hsfield = copy(p.hsfield)

  global_update_perform_shift!(s,p,l)
  build_stack(s, p, l)
  propagate(s, p, l)
  # now we have s.greens = G_{p.slices} (up to one down-wrap) and s.log_det = logdet of G_{p.slices + 1} for new globally shifted conf

  p.boson_action = calculate_boson_action(p, l)

  p_boson = exp(-(p.boson_action - S_old)) # exp_delta_S_boson

  # calculate detratio = fermion accept. prob.
  p_fermion = exp(s.gb_log_det - s.log_det)

  p_acc = p_boson * real(p_fermion)

  # @printf("p_boson %.2e\n",abs.(p_boson))
  # @printf("p_fermion %.2e\n",abs.(p_fermion))
  # @printf("p_acc %.2e\n",abs.(p_acc))
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