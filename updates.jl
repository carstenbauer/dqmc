function local_updates(s::stack, p::parameters, l::lattice)
  acc_rat = 0.0
  @inbounds for i in 1:l.sites
    new_op = rand(p.box, 3)
    exp_delta_S_boson = exp(-boson_action_diff(s,p,l,i,new_op))
    detratio = calculate_detratio(s,p,l,i,new_op)
    # double detratioAbsSq = abs(detratio) * abs(detratio)
    p_acc = min(1.0,exp_delta_S_boson * det_ratio)

    if rand() < p_acc
      acc_rat += 1
      p.hsfield[:,i,s.current_slice] = new_op
      p.boson_action += -log(exp_delta_S_boson)
      update_greens(s,p,l,i,new_op)
    end
  end
  return acc_rat / l.sites
end


function calculate_detratio(s::stack, p::parameters, l::lattice, i::Int, new_op::Vector{Float64})
  # TODO calculate_detratio in constant time
  return 0.0 + 0.0im
end


function update_greens(s::stack, p::parameters, l::lattice)
  # TODO update_greens after local update
  nothing
end


# function simple_update(s::stack, p::parameters, l::lattice)
#   u = zeros(Complex{Float64}, l.n_sites)
#   v_dag = zeros(Complex{Float64}, l.n_sites)
#   iden = eye(Complex{Float64}, l.n_sites)
#   spins_flipped = 0
#   # println("Updating ", s.current_slice)
#   rel_phase = 1. + 0im
#   @inbounds for i in 1:l.n_sites
#     gamma = exp(1. * 2 * p.af_field[i, s.current_slice] * p.lambda) - 1
#
#     prob = (1 + gamma * (1 - s.greens[i,i]))^2 / (gamma + 1)
#
#     if p.particles == Int(0.5 * l.n_sites) && abs(imag(prob)) > 1e-7
#       println("Did you expect a sign problem?", abs(imag(prob)))
#       @printf "%.10e" abs(imag(prob))
#     end
#
#     if rand() < abs(prob) / (1. + abs(prob))
#         rel_phase *= prob / abs(prob)
#       u[:] = -s.greens[:, i]
#       u[i] += 1.
#       # v_dag[i] = gamma
#
#       # s.greens = (iden - kron(u * 1./(1 + vecdot(conj(v_dag), u)), transpose(v_dag))) * s.greens
#       # s.greens -= kron(u * 1./(1 + vecdot(conj(v_dag), u)), gamma * s.greens[i, :])
#       s.greens -= kron(u * 1./(1 + gamma * u[i]), gamma * s.greens[i, :])
#       p.af_field[i, s.current_slice] *= -1.
#       spins_flipped += 1
#       # v_dag[i] *= 0
#     end
#   end
#   return spins_flipped / l.n_sites, rel_phase
# end
