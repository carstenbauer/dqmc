# interaction_matrix_exp = exp(- pref delta_tau V(slice))
function interaction_matrix_exp(p::Parameters, l::Lattice, slice::Int, pref::Float64=1.)
  C = zeros(l.sites,l.sites)
  S = zeros(Complex{Float64}, l.sites,l.sites)
  R = zeros(l.sites,l.sites)
  for i in 1:l.sites
    sh = sinh(pref * p.lambda * p.delta_tau * norm(p.hsfield[:,i,slice]))/norm(p.hsfield[:,i,slice])
    C[i,i] = cosh(pref * p.lambda * p.delta_tau * norm(p.hsfield[:,i,slice]))
    S[i,i] = (im * p.hsfield[2,i,slice] - p.hsfield[1,i,slice]) * sh
    R[i,i] = (-p.hsfield[3,i,slice]) * sh
  end
  Z = zeros(l.sites,l.sites)

  return [C S Z R; conj(S) C -R Z; Z -R C conj(S); R Z S C]
end
# SPEED: Implementation that sets all matrix element explicitly. Is it really more efficient?

# calculate p.flv x p.flv (4x4 for O(3) model) interaction matrix exponential for given op
function interaction_matrix_exp_op(p::Parameters, l::Lattice, op::Vector{Float64}, pref::Float64=1.)
  sh = sinh(pref * p.lambda * p.delta_tau*norm(op))/norm(op)
  Cii = cosh(pref * p.lambda * p.delta_tau*norm(op))
  Sii = (im * op[2] - op[1]) * sh
  Rii = (-op[3]) * sh

  return [Cii Sii 0 Rii; conj(Sii) Cii -Rii 0; 0 -Rii Cii conj(Sii); Rii 0 Sii Cii]
end
























# function interaction_matrix_left!(M::Array{Complex{Float64}, 2}, p::Parameters, l::Lattice, slice, pref::Float64=1.)
#   for i in 1:l.n_sites
#     M[i, :] *= exp(-pref * p.lambda * p.af_field[i, slice] - pref * p.delta_tau * p.mu)
#   end
# end
#
# function interaction_matrix_right!(M::Array{Complex{Float64}, 2}, p::Parameters, l::Lattice, slice, pref::Float64=1.)
#   for i in 1:l.n_sites
#     M[:, i] *= exp(-pref * p.lambda * p.af_field[i, slice] - pref * p.delta_tau * p.mu)
#   end
# end
