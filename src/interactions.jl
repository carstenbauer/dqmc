@inline function setblockdiag!(l::Lattice, A::AbstractSparseMatrix, row::Int, col::Int, B::AbstractVector)
  rstart = (row-1)*l.sites+1
  colstart = (col-1)*l.sites+1

  @inbounds for shift in 0:length(B)-1
    A[rstart+shift,colstart+shift] = B[shift+1]
  end
  nothing
end

@inline function setblockdiag_minus!(l::Lattice, A::AbstractSparseMatrix, row::Int, col::Int, B::AbstractVector)
  rstart = (row-1)*l.sites+1
  colstart = (col-1)*l.sites+1

  @inbounds for shift in 0:length(B)-1
    A[rstart+shift,colstart+shift] = -B[shift+1] # minus here!
  end
  nothing
end





# interaction_matrix_exp = exp(- power delta_tau V(slice)), with power = +- 1.
function interaction_matrix_exp(mc::AbstractDQMC, slice::Int, power::Float64=1.)
  G = geltype(mc)

  eV = spzeros(G, mc.p.flv * mc.l.sites, mc.p.flv * mc.l.sites)
  interaction_matrix_exp!(mc, slice, power, eV)
  return eV
end

# interaction_matrix_exp = exp(- power delta_tau V(slice)), with power = +- 1.
function interaction_matrix_exp!(mc::AbstractDQMC, slice::Int, power::Float64=1., eV::AbstractSparseMatrix=mc.g.eV)
  @mytimeit mc.a.to "interactions (combined)" begin
  @mytimeit mc.a.to "interaction_matrix_exp!" begin
  p = mc.p
  l = mc.l
  N = l.sites
  G = geltype(mc)

  C = mc.g.C
  S = mc.g.S
  R = mc.g.R

  # C = zeros(G, N) #1, 1)
  # S = zeros(G, N) #1, 2)
  # if p.opdim == 3
    # R = zeros(G, N) #1, 4)
  # end
  @inbounds @simd for i in 1:N
    @views n = norm(p.hsfield[:,i,slice])
    sh = sinh(p.lambda * p.delta_tau * n)/n
    C[i] = cosh(p.lambda * p.delta_tau * n)
    if p.opdim == 3
      S[i] = (im * p.hsfield[2,i,slice] - p.hsfield[1,i,slice]) * power * sh
      R[i] = (-p.hsfield[3,i,slice]) * power * sh
    elseif p.opdim == 2
      S[i] = (im * p.hsfield[2,i,slice] - p.hsfield[1,i,slice]) * power * sh
    else # O(1)
      S[i] = - p.hsfield[1,i,slice] * power * sh
    end
  end

  setblockdiag!(l,eV,1,1,C)
  setblockdiag!(l,eV,1,2,S)
  
  cS = conj(S) # Once julia 1.2 is out, use lazy ConjVector here.
  setblockdiag!(l,eV,2,1,cS)
  setblockdiag!(l,eV,2,2,C)

  if p.opdim == 3
    setblockdiag!(l,eV,1,4,R)
    
    setblockdiag_minus!(l,eV,2,3,R)

    setblockdiag_minus!(l,eV,3,2,R)
    setblockdiag!(l,eV,3,3,C)
    setblockdiag!(l,eV,3,4,cS)

    setblockdiag!(l,eV,4,1,R)
    setblockdiag!(l,eV,4,3,S)
    setblockdiag!(l,eV,4,4,C)
  end
  end #timeit
  end #timeit
end





# calculate p.flv x p.flv (4x4 for O(3) model) interaction matrix exponential for given op
function interaction_matrix_exp_op(mc::AbstractDQMC, op::Vector{Float64}, power::Float64=1.)
  eVop = Matrix{geltype(mc)}(undef,mc.p.flv,mc.p.flv)
  interaction_matrix_exp_op!(mc,op,power,eVop)
  return eVop
end

# calculate p.flv x p.flv (4x4 for O(3) model) interaction matrix exponential for given op
function interaction_matrix_exp_op!(mc::AbstractDQMC{C,G}, op::Vector{Float64}, power::Float64=1., eVop::Matrix{G}=mc.g.eVop1) where {C,G}
  @mytimeit mc.a.to "interactions (combined)" begin
  @mytimeit mc.a.to "interaction_matrix_exp_op!" begin
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

  if mc.p.opdim == 3
    eVop[1,3] = zero(G)
    eVop[1,4] = Rii

    eVop[2,3] = -Rii
    eVop[2,4] = zero(G)
    
    eVop[3,1] = zero(G)
    eVop[3,2] = -Rii
    eVop[3,3] = Cii
    eVop[3,4] = conj(Sii)
    
    eVop[4,1] = Rii
    eVop[4,2] = zero(G)
    eVop[4,3] = Sii
    eVop[4,4] = Cii
  end
  end #timeit
  end #timeit
end