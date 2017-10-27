# interaction_matrix_exp = exp(- power delta_tau V(slice)), with power = +- 1.
function interaction_matrix_exp(p::Parameters, l::Lattice, slice::Int, power::Float64=1.)
  eV = zeros(Complex{Float64}, p.flv * l.sites, p.flv * l.sites)

  C = blockview(l, eV, 1, 1)
  S = blockview(l, eV, 1, 2)
  R = blockview(l, eV, 1, 4)
  @simd for i in 1:l.sites
    n = norm(p.hsfield[:,i,slice])
    sh = sinh(p.lambda * p.delta_tau * n)/n
    C[i,i] = cosh(p.lambda * p.delta_tau * n)
    S[i,i] = (im * p.hsfield[2,i,slice] - p.hsfield[1,i,slice]) * power * sh
    R[i,i] = (-p.hsfield[3,i,slice]) * power * sh
  end

  cS = conj(S)
  mR = -R
  blockreplace!(l,eV,2,1,cS)
  blockreplace!(l,eV,2,2,C)
  blockreplace!(l,eV,2,3,mR)

  blockreplace!(l,eV,3,2,mR)
  blockreplace!(l,eV,3,3,C)
  blockreplace!(l,eV,3,4,cS)

  blockreplace!(l,eV,4,1,R)
  blockreplace!(l,eV,4,3,S)
  blockreplace!(l,eV,4,4,C)

  return eV
end

blockview{T<:Number}(l::Lattice, A::Matrix{T}, row::Int, col::Int) = view(A, (row-1)*l.sites+1:row*l.sites, (col-1)*l.sites+1:col*l.sites)
function blockreplace!{T<:Number}(l::Lattice, A::Matrix{T}, row::Int, col::Int, B::Union{Matrix{T},SubArray{T,2}})
  @inbounds A[(row-1)*l.sites+1:row*l.sites, (col-1)*l.sites+1:col*l.sites] = B
  nothing
end

# calculate p.flv x p.flv (4x4 for O(3) model) interaction matrix exponential for given op
function interaction_matrix_exp_op(p::Parameters, l::Lattice, op::Vector{Float64}, power::Float64=1.)
  n = norm(op)
  sh = power * sinh(p.lambda * p.delta_tau*n)/n
  Cii = cosh(p.lambda * p.delta_tau*n)
  Sii = (im * op[2] - op[1]) * sh
  Rii = (-op[3]) * sh

  r = Matrix{Complex128}(4,4)
  
  r[1,1] = Cii
  r[1,2] = Sii
  r[1,3] = zero(Complex128)
  r[1,4] = Rii
  
  r[2,1] = conj(Sii)
  r[2,2] = Cii
  r[2,3] = -Rii
  r[2,4] = zero(Complex128)
  
  r[3,1] = zero(Complex128)
  r[3,2] = -Rii
  r[3,3] = Cii
  r[3,4] = conj(Sii)
  
  r[4,1] = Rii
  r[4,2] = zero(Complex128)
  r[4,3] = Sii
  r[4,4] = Cii

  return r
end


function interaction_matrix_slow(p::Parameters, l::Lattice, slice::Int, power::Float64=1.)
  C = zeros(l.sites,l.sites)
  S = zeros(Complex{Float64}, l.sites,l.sites)
  R = zeros(l.sites,l.sites)
  for i in 1:l.sites
    sh = power * sinh(p.lambda * p.delta_tau * norm(p.hsfield[:,i,slice]))/norm(p.hsfield[:,i,slice])
    C[i,i] = cosh(p.lambda * p.delta_tau * norm(p.hsfield[:,i,slice]))
    S[i,i] = (im * p.hsfield[2,i,slice] - p.hsfield[1,i,slice]) * sh
    R[i,i] = (-p.hsfield[3,i,slice]) * sh
  end
  Z = zeros(l.sites,l.sites)

  return [C S Z R; conj(S) C -R Z; Z -R C conj(S); R Z S C]
end