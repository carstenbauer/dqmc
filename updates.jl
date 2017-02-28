function local_updates(s::Stack, p::Parameters, l::Lattice)
  acc_rat = 0.0
  @inbounds for i in 1:l.sites
    new_op = p.hsfield[:,i,s.current_slice] + rand(p.box, 3)
    exp_delta_S_boson = exp(-calculate_boson_action_diff(p,l,i,new_op))
    detratio = calculate_detratio(s,p,l,i,new_op)

    # if abs(imag(detratio)) > 1e-3 || abs(imag(detratio))/abs(real(detratio)) > 1e-3
      # @printf("%d, %d \t Determinant ratio isn't real. \t abs imag: %.3e \t relative: %.1f%%\n", s.current_slice, i, abs(imag(detratio)), abs(imag(detratio))/abs(real(detratio))*100)
    # elseif real(detratio) < 0
      # println("Negative fermion weight.")
    # elseif detratio == 0
      # println("Encountered non-invertible M with det = 0.")
    # end

    p_acc = exp_delta_S_boson * real(detratio)
    # println("p_acc: ",p_acc)
    # println("fermion prob: ",detratio)
    # println("boson prob: ",exp_delta_S_boson)

    if p_acc > 1.0 || rand() < p_acc
      acc_rat += 1
      p.hsfield[:,i,s.current_slice] = new_op[:]
      p.boson_action += -log(exp_delta_S_boson)
      update_interaction_sinh_cosh(p,l,i,s.current_slice,new_op)
      update_greens!(s,p,l,i)
    end
  end
  return acc_rat / l.sites
end


function calculate_detratio(s::Stack, p::Parameters, l::Lattice, i::Int, new_op::Vector{Float64})
  # TODO: Compare delta_i and M calculation (detratio)
  # TODO: Why is detratio not completely real??
  V1i = interaction_matrix_exp_op(p,l,p.hsfield[:,i,s.current_slice],-1.)
  V2i = interaction_matrix_exp_op(p,l,new_op)
  s.delta_i = V1i * V2i  - s.eye_flv
  s.M = s.eye_flv + s.delta_i * (s.eye_flv - s.greens[i:l.sites:end,i:l.sites:end])
  return det(s.M)
end


function update_greens!(s::Stack, p::Parameters, l::Lattice, i::Int)
  # first_term = (s.greens - s.eye_full)[:,i:l.sites:end] * inv(s.M)
  first_term = /((s.greens - s.eye_full)[:,i:l.sites:end], s.M)
  second_term = s.delta_i * s.greens[i:l.sites:end,:]
  s.greens = s.greens + first_term * second_term
end


function global_update(s::Stack, p::Parameters, l::Lattice)
  global_op_shift = rand(p.box, 3)
  new_hsfield = mapslices(x -> x + global_op_shift, p.hsfield, [1])
  S_new = calculate_boson_action(p, l, new_hsfield)
  
  exp_delta_S_boson = exp(-(S_new - p.boson_action))
  # detratio = ?
  p_acc = 0.0

  if p_acc > 1.0 || rand() < p_acc
      p.hsfield[:] = new_hsfield[:]
      p.boson_action = S_new
      init_interaction_sinh_cosh(p,l)
      # s.greens = ?
      return true
  end
  return false
end