function local_updates(s::Stack, p::Parameters, l::Lattice)
  acc_rat = 0.0
  @inbounds for i in 1:l.sites
    @views new_op = p.hsfield[:,i,s.current_slice] + rand(p.box, 3)
    exp_delta_S_boson = exp(- calculate_boson_action_diff(p, l, i, s.current_slice, new_op) )
    detratio = calculate_detratio(s,p,l,i,new_op)

    if abs(imag(detratio)/real(detratio)) > 1e-4
      @printf("%d, %d \t Determinant ratio isn't real. \t abs imag: %.3e \t relative: %.1f%%\n", s.current_slice, i, abs.(imag(detratio)), abs.(imag(detratio))/abs.(real(detratio))*100)
    # elseif real(detratio) < 0
    #   println("Negative fermion weight.")
    # elseif detratio == 0
    #   println("Encountered non-invertible M with det = 0.")
    end

    p_acc = exp_delta_S_boson * real(detratio)

    if p_acc > 1.0 || rand() < p_acc
      acc_rat += 1
      @views p.hsfield[:,i,s.current_slice] = new_op
      p.boson_action += -log(exp_delta_S_boson)
      update_greens!(s,p,l,i)
    end
  end
  return acc_rat / l.sites
end


@inline function calculate_detratio(s::Stack, p::Parameters, l::Lattice, i::Int, new_op::Vector{Float64})
  interaction_matrix_exp_op!(p,l,p.hsfield[:,i,s.current_slice],-1.,p.eVop1) #V1i
  interaction_matrix_exp_op!(p,l,new_op,1.,p.eVop2) #V2i
  s.delta_i = p.eVop1 * p.eVop2  - s.eye_flv
  s.M = s.eye_flv + s.delta_i * (s.eye_flv - s.greens[i:l.sites:end,i:l.sites:end])
  return det(s.M)
end

@inline function update_greens!(s::Stack, p::Parameters, l::Lattice, i::Int)
  A = s.greens[:,i:l.sites:end]
  @simd for k in 0:3
      A[i+k*l.sites,k+1] -= 1.
  end
  A *= inv(s.M)
  B = s.delta_i * s.greens[i:l.sites:end,:]
  s.greens .+= A * B
end