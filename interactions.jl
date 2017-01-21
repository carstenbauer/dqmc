# interaction_matrix = matrix exponential of V = exp(- pref delta_tau V(slice))
function interaction_matrix(p::parameters, l::lattice, slice, pref::Float64=1.)
  C = zeros(l.sites,l.sites)
  S = zeros(Complex{Float64}, l.sites,l.sites)
  R = zeros(l.sites,l.sites)
  for i in 1:l.sites
    sh = sinh(pref * p.delta_tau*norm(p.hsfield[:,i,slice]))/norm(p.hsfield[:,i,slice])
    C[i,i] = p.lambda * cosh(pref * p.delta_tau*norm(p.hsfield[:,i,slice]))
    S[i,i] = p.lambda * (im * p.hsfield[2,i,slice] - p.hsfield[1,i,slice]) * sh
    R[i,i] = p.lambda * (-p.hsfield[3,i,slice]) * sh
  end
  Z = zeros(l.sites,l.sites)

  return [C Z R S; Z C conj(S) -R; R S C Z; conj(S) -R Z C]
end
























# function interaction_matrix_left!(M::Array{Complex{Float64}, 2}, p::parameters, l::lattice, slice, pref::Float64=1.)
#   for i in 1:l.n_sites
#     M[i, :] *= exp(-pref * p.lambda * p.af_field[i, slice] - pref * p.delta_tau * p.mu)
#   end
# end
#
# function interaction_matrix_right!(M::Array{Complex{Float64}, 2}, p::parameters, l::lattice, slice, pref::Float64=1.)
#   for i in 1:l.n_sites
#     M[:, i] *= exp(-pref * p.lambda * p.af_field[i, slice] - pref * p.delta_tau * p.mu)
#   end
# end
