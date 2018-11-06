function global_update_backup_swap!(mc::AbstractDQMC)
  # swap current stack and greens to backup stack and greens (more efficient than copying)
  mc.s.gb_u_stack, mc.s.u_stack = mc.s.u_stack, mc.s.gb_u_stack
  mc.s.gb_d_stack, mc.s.d_stack = mc.s.d_stack, mc.s.gb_d_stack
  mc.s.gb_t_stack, mc.s.t_stack = mc.s.t_stack, mc.s.gb_t_stack
  mc.s.gb_greens, mc.s.greens = mc.s.greens, mc.s.gb_greens # this is greens at time slice == p.slices
  mc.s.gb_log_det, mc.s.log_det = mc.s.log_det, mc.s.gb_log_det # this is logdet of greens at time slice == p.slices + 1
end

@inline function global_update_perform_shift!(mc::AbstractDQMC)
    @inbounds @views begin
      @simd for k in 1:mc.p.opdim
        mc.p.hsfield[k,:,:] .+= rand(mc.p.box_global)
      end
    end
end

function global_update(mc::AbstractDQMC)
  p = mc.p
  s = mc.s

  @assert (s.current_slice == p.slices) && (s.direction == -1)

  S_old = p.boson_action
  # if p.all_checks && !isapprox(S_old,calc_boson_action(p, l)) warn("Incorrect boson action found during attempt to do global update.") end

  global_update_backup_swap!(mc) # save current stack etc.
  s.gb_hsfield = copy(p.hsfield)

  global_update_perform_shift!(mc)
  build_stack(mc)
  propagate(mc)
  # now we have s.greens = G_{p.slices} (up to one down-wrap) and s.log_det = logdet of G_{p.slices + 1} for new globally shifted conf

  p.boson_action = calc_boson_action(mc)

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
    global_update_backup_swap!(mc)
    return 0
  end

end