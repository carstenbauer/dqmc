function local_updates(mc::DQMC)
  const p = mc.p
  const s = mc.s
  const l = mc.l

  acc_rat = 0.0
  @inbounds for i in 1:l.sites
    @views new_op = p.hsfield[:,i,s.current_slice] + rand(p.box, p.opdim)
    exp_delta_S_boson = exp(- calculate_boson_action_diff(mc, i, s.current_slice, new_op) )
    detratio = calculate_detratio(mc,i,new_op)

    if p.opdim == 3
      p_acc_fermion = real(detratio)
    elseif p.opdim < 3
      p_acc_fermion = real(detratio * conj(detratio))
    end

    if p.opdim == 3 && abs(imag(detratio)/real(detratio)) > 1e-4
      @printf("%d, %d \t Determinant ratio isn't real. \t abs imag: %.3e \t relative: %.1f%%\n", s.current_slice, i, abs.(imag(detratio)), abs.(imag(detratio))/abs.(real(detratio))*100)
    # elseif real(detratio) < 0
    #   println("Negative fermion weight.")
    # elseif detratio == 0
    #   println("Encountered non-invertible M with det = 0.")
    elseif p.opdim < 3 && p_acc_fermion < 0
      println("Negative fermion weight!")
    end

    p_acc = exp_delta_S_boson * p_acc_fermion

    if p_acc > 1.0 || rand() < p_acc
      acc_rat += 1
      @views p.hsfield[:,i,s.current_slice] = new_op
      p.boson_action += -log(exp_delta_S_boson)
      update_greens!(mc,i)
    end
  end
  return acc_rat / l.sites
end


@inline function calculate_detratio(mc::DQMC, i::Int, new_op::Vector{Float64})
  const p = mc.p
  const s = mc.s
  const l = mc.l

  interaction_matrix_exp_op!(mc,p.hsfield[:,i,s.current_slice],-1.,s.eVop1) #V1i
  interaction_matrix_exp_op!(mc,new_op,1.,s.eVop2) #V2i
  s.delta_i = s.eVop1 * s.eVop2  - eye_flv
  s.M = eye_flv + s.delta_i * (eye_flv - s.greens[i:l.sites:end,i:l.sites:end])
  return det(s.M)
end

@inline function update_greens!(mc::DQMC, i::Int)
  const p = mc.p
  const s = mc.s
  const l = mc.l

  A = s.greens[:,i:l.sites:end]
  
  if p.opdim == 3
    @simd for k in 0:3
        A[i+k*l.sites,k+1] -= 1.
    end
  else
    @simd for k in 0:1
        A[i+k*l.sites,k+1] -= 1.
    end
  end

  A *= inv(s.M)
  B = s.delta_i * s.greens[i:l.sites:end,:]
  s.greens .+= A * B
end