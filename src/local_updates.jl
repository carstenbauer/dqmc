function local_updates(mc::AbstractDQMC)
  p = mc.p
  g = mc.g
  l = mc.l
  a = mc.a

  acc_rat = 0.0
  @inbounds for i in 1:l.sites
    @views new_op = p.hsfield[:,i,g.current_slice] + randuniform(p.box, p.opdim)
    exp_delta_S_boson = exp(- calc_boson_action_diff(mc, i, g.current_slice, new_op) )
    detratio = calc_detratio(mc,i,new_op)

    if p.opdim == 3
      p_acc_fermion = real(detratio)
    elseif p.opdim < 3
      p_acc_fermion = real(detratio * conj(detratio))
    end

    if p.opdim == 3 && abs(imag(detratio)/real(detratio)) > 1e-4
      @printf("%d, %d \t Determinant ratio isn't real. \t abs imag: %.3e \t relative: %.1f%%\n", g.current_slice, i, abs.(imag(detratio)), abs.(imag(detratio))/abs.(real(detratio))*100)
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
      @views p.hsfield[:,i,g.current_slice] = new_op
      p.boson_action += -log(exp_delta_S_boson)
      update_greens!(mc,i)
    end
  end
  return acc_rat / l.sites
end


@inline function calc_detratio(mc::AbstractDQMC, i::Int, new_op::Vector{Float64})
  p = mc.p
  g = mc.g
  s = mc.s
  l = mc.l

  @mytimeit mc.a.to "calc_detratio" begin

  interaction_matrix_exp_op!(mc,p.hsfield[:,i,g.current_slice],-1.,s.eVop1) #V1i
  interaction_matrix_exp_op!(mc,new_op,1.,s.eVop2) #V2i
  mul!(s.eVop1eVop2, s.eVop1, s.eVop2)
  s.delta_i .= s.eVop1eVop2 .- s.eye_flv
  s.Mtmp .= s.eye_flv .- g.greens[i:l.sites:end,i:l.sites:end]
  mul!(s.Mtmp2, s.delta_i, s.Mtmp)
  s.M .= s.eye_flv .+ s.Mtmp2

  end #timeit
  return det(s.M)
end

@inline function update_greens!(mc::AbstractDQMC, i::Int)
  p = mc.p
  g = mc.g
  s = mc.s
  l = mc.l
  ab = mc.s.AB

  @mytimeit mc.a.to "update_greens!" begin

  s.A = g.greens[:,i:l.sites:end]
  
  if p.opdim == 3
    @simd for k in 0:3
        s.A[i+k*l.sites,k+1] -= 1.
    end
  else
    @simd for k in 0:1
        s.A[i+k*l.sites,k+1] -= 1.
    end
  end

  s.A *= inv(s.M)
  mul!(s.B, s.delta_i, g.greens[i:l.sites:end,:])

  # benchmark: most of time (>90% is spend below)
  mul!(s.AB, s.A, s.B)
  
  # more explicit way of doing g.greens .+= s.AB
  @inbounds @simd for i in eachindex(g.greens)
    g.greens[i] += ab[i]
  end

  end #timeit
  nothing
end