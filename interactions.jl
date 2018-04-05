if !isdefined(:GreensType)
  global const GreensType = Complex128; # assume O(2) or O(3)
  warn("GreensType wasn't set on loading interactions.jl")
  println("GreensType = ", GreensType)
end

# interaction_matrix_exp = exp(- power delta_tau V(slice)), with power = +- 1.
function interaction_matrix_exp(mc::AbstractDQMC, slice::Int, power::Float64=1.)
  eV = zeros(GreensType, mc.p.flv * mc.l.sites, mc.p.flv * mc.l.sites)
  interaction_matrix_exp!(mc, slice, power, eV)
  return eV
end

# interaction_matrix_exp = exp(- power delta_tau V(slice)), with power = +- 1.
function interaction_matrix_exp!(mc::AbstractDQMC, slice::Int, power::Float64=1., eV::Matrix{GreensType}=mc.s.eV)
  const p = mc.p
  const l = mc.l

  C = blockview(l, eV, 1, 1)
  S = blockview(l, eV, 1, 2)
  if p.opdim == 3
    R = blockview(l, eV, 1, 4)
  end
  @simd for i in 1:l.sites
    n = norm(p.hsfield[:,i,slice])
    sh = sinh(p.lambda * p.delta_tau * n)/n
    C[i,i] = cosh(p.lambda * p.delta_tau * n)
    if p.opdim == 3
      S[i,i] = (im * p.hsfield[2,i,slice] - p.hsfield[1,i,slice]) * power * sh
      R[i,i] = (-p.hsfield[3,i,slice]) * power * sh
    elseif p.opdim == 2
      S[i,i] = (im * p.hsfield[2,i,slice] - p.hsfield[1,i,slice]) * power * sh
    else # O(1)
      S[i,i] = - p.hsfield[1,i,slice] * power * sh
    end
  end

  cS = conj(S)
  blockreplace!(l,eV,2,1,cS)
  blockreplace!(l,eV,2,2,C)

  if p.opdim == 3
    mR = -R
    blockreplace!(l,eV,2,3,mR)

    blockreplace!(l,eV,3,2,mR)
    blockreplace!(l,eV,3,3,C)
    blockreplace!(l,eV,3,4,cS)

    blockreplace!(l,eV,4,1,R)
    blockreplace!(l,eV,4,3,S)
    blockreplace!(l,eV,4,4,C)
  end
end

@inline blockview(l::Lattice, A::AbstractMatrix{T}, row::Int, col::Int) where T<:Number = view(A, (row-1)*l.sites+1:row*l.sites, (col-1)*l.sites+1:col*l.sites)
@inline function blockreplace!(l::Lattice, A::AbstractMatrix{T}, row::Int, col::Int, B::AbstractMatrix{T}) where T<:Number
  @views A[(row-1)*l.sites+1:row*l.sites, (col-1)*l.sites+1:col*l.sites] = B
  nothing
end

# calculate p.flv x p.flv (4x4 for O(3) model) interaction matrix exponential for given op
function interaction_matrix_exp_op(mc::AbstractDQMC, op::Vector{Float64}, power::Float64=1.)
  eVop = Matrix{GreensType}(mc.p.flv,mc.p.flv)
  interaction_matrix_exp_op!(mc,op,power,eVop)
  return eVop
end

# calculate p.flv x p.flv (4x4 for O(3) model) interaction matrix exponential for given op
function interaction_matrix_exp_op!(mc::AbstractDQMC, op::Vector{Float64}, power::Float64=1., eVop::Matrix{GreensType}=mc.s.eVop1)
  n = norm(op)
  sh = power * sinh(mc.p.lambda * mc.p.delta_tau*n)/n
  Cii = cosh(mc.p.lambda * mc.p.delta_tau*n)
  if mc.p.opdim == 3
    Sii = (im * op[2] - op[1]) * sh
    Rii = (-op[3]) * sh
  elseif mc.p.opdim == 2
    Sii = (im * op[2] - op[1]) * sh
  else # O(1)
    Sii = - op[1] * sh
  end
  
  eVop[1,1] = Cii
  eVop[1,2] = Sii  
  eVop[2,1] = conj(Sii)
  eVop[2,2] = Cii

  if p.opdim == 3
    eVop[1,3] = zero(GreensType)
    eVop[1,4] = Rii

    eVop[2,3] = -Rii
    eVop[2,4] = zero(GreensType)
    
    eVop[3,1] = zero(GreensType)
    eVop[3,2] = -Rii
    eVop[3,3] = Cii
    eVop[3,4] = conj(Sii)
    
    eVop[4,1] = Rii
    eVop[4,2] = zero(GreensType)
    eVop[4,3] = Sii
    eVop[4,4] = Cii
  end
end