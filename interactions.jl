# interaction_matrix_exp = exp(- power delta_tau V(slice)), with power = +- 1.
function interaction_matrix_exp(p::Parameters, l::Lattice, slice::Int, power::Float64=1.)
  eV = zeros(Complex{Float64}, p.flv * l.sites, p.flv * l.sites)
  interaction_matrix_exp!(p, l, slice, power, eV)
  return eV
end

# interaction_matrix_exp = exp(- power delta_tau V(slice)), with power = +- 1.
function interaction_matrix_exp!(p::Parameters, l::Lattice, slice::Int, power::Float64=1., eV::Matrix{Complex128}=p.eV)

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

  println("typeofC ", typeof(C))
  println("typeofS ", typeof(S))
  println("typeofR ", typeof(R))

  cS = conj(S)
  mR = -R
  println("typeofcS ", typeof(cS))
  println("typeofmR ", typeof(mR))
  blockreplace!(l,eV,2,1,cS)
  blockreplace!(l,eV,2,2,C)
  blockreplace!(l,eV,2,3,mR)

  blockreplace!(l,eV,3,2,mR)
  blockreplace!(l,eV,3,3,C)
  blockreplace!(l,eV,3,4,cS)

  blockreplace!(l,eV,4,1,R)
  blockreplace!(l,eV,4,3,S)
  blockreplace!(l,eV,4,4,C)
end

blockview(l::Lattice, A::AbstractMatrix, row::Int, col::Int) = view(A, (row-1)*l.sites+1:row*l.sites, (col-1)*l.sites+1:col*l.sites)
function blockreplace!(l::Lattice, A::AbstractMatrix, row::Int, col::Int, B::AbstractMatrix)
  println(typeof(B))
  @views A[(row-1)*l.sites+1:row*l.sites, (col-1)*l.sites+1:col*l.sites] = B
  nothing
end

# calculate p.flv x p.flv (4x4 for O(3) model) interaction matrix exponential for given op
function interaction_matrix_exp_op(p::Parameters, l::Lattice, op::Vector{Float64}, power::Float64=1.)

  eVop = Matrix{Complex128}(4,4)
  interaction_matrix_exp_op!(p,l,op,power,eVop)
  return eVop
end

# calculate p.flv x p.flv (4x4 for O(3) model) interaction matrix exponential for given op
function interaction_matrix_exp_op!(p::Parameters, l::Lattice, op::Vector{Float64}, power::Float64=1., eVop::Matrix{Complex128}=p.eVop1)
  n = norm(op)
  sh = power * sinh(p.lambda * p.delta_tau*n)/n
  Cii = cosh(p.lambda * p.delta_tau*n)
  Sii = (im * op[2] - op[1]) * sh
  Rii = (-op[3]) * sh
  
  eVop[1,1] = Cii
  eVop[1,2] = Sii
  eVop[1,3] = zero(Complex128)
  eVop[1,4] = Rii
  
  eVop[2,1] = conj(Sii)
  eVop[2,2] = Cii
  eVop[2,3] = -Rii
  eVop[2,4] = zero(Complex128)
  
  eVop[3,1] = zero(Complex128)
  eVop[3,2] = -Rii
  eVop[3,3] = Cii
  eVop[3,4] = conj(Sii)
  
  eVop[4,1] = Rii
  eVop[4,2] = zero(Complex128)
  eVop[4,3] = Sii
  eVop[4,4] = Cii
end