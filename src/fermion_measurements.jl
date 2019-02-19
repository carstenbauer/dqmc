# -------------------------------------------------------
#         Measurements: Greens (ETGF)
# -------------------------------------------------------
function measure_greens_and_logdet(mc::AbstractDQMC, safe_mult::Int=mc.p.safe_mult)
  greens, greens_logdet = calc_greens_and_logdet(mc, 1, safe_mult)
  effective_greens2greens!(mc, greens)
  return greens, greens_logdet
end

function measure_greens(mc::AbstractDQMC, safe_mult::Int=mc.p.safe_mult)
  return measure_greens_and_logdet(mc, safe_mult)[1]
end



# -------------------------------------------------------
#         Occupation
# -------------------------------------------------------
"""
Overall occupation, averaged over sites and fermion flavors (both spin and band).

Example: half-filling of spin 1/2 fermions on a lattice corresponds to `n=0.5`, not `n=1`.
"""
function occupation(mc::AbstractDQMC, greens::AbstractMatrix=mc.s.greens)
  n = mean(1 .- diag(greens))
  # only per site:
  # N = mc.l.sites
  # M = mc.p.slices
  # n = 1/(N*M) * sum(1 .- diag(greens))
  real(n)
end

"""
Flavor resolved occupations (only averaged over sites).
"""
function occupations_flv(mc::AbstractDQMC, greens::AbstractMatrix=mc.s.greens)
  N = mc.l.sites
  ns = mean.(Iterators.partition(1 .- diag(greens), N))
  # only per site:
  # N = mc.l.sites
  # M = mc.p.slices
  # n = 1/(N*M) * sum(1 .- diag(greens))
  real(ns)
end




# -------------------------------------------------------
#         Equal time pairing correlations
# -------------------------------------------------------
function allocate_etpc!(mc)
  L = mc.p.L
  meas = mc.s.meas

  meas.etpc_evs = zeros(Float64,4,L,L)
  meas.etpc_minus = zeros(Float64,L,L)
  meas.etpc_plus = zeros(Float64,L,L)

  println("Allocated memory for ETPC measurements.")
  nothing
end

"""
Calculate equal time pairing susceptibilities

  Pη(y,x) = << Δη^†(r)Δη(0) >> = etpc_plus/minus(y,x)

where η=± (s and d-wave), r=(y,x), and <<·>> indicates a fermion average for fixed ϕ.

x,y ∈ [0,L-1]

Details:
- we mean over r_0, that is "0".
- we do not mean over τ_0=0 (τ not shown above).
"""
function etpc!(mc::AbstractDQMC, greens::AbstractMatrix)
  L = mc.p.L
  G = geltype(mc)
  ev = mc.s.meas.etpc_evs
  Pm = mc.s.meas.etpc_minus # d-wave
  Pp = mc.s.meas.etpc_plus # s-wave

  # fill!(Pm, 0.)
  # fill!(Pp, 0.)
  etpc_evs!(mc, greens)

  # xd,xu,xu,xd = 1
  # xd,xu,yu,yd = 2
  # yd,yu,xu,xd = 3
  # yd,yu,yu,yd = 4

  # fill result arrays
  @inbounds for x in 1:L
    @simd for y in 1:L
      Pm[y,x] = real(ev[1,y,x] - ev[2,y,x] - ev[3,y,x] + ev[4,y,x])
      Pp[y,x] = real(ev[1,y,x] + ev[2,y,x] + ev[3,y,x] + ev[4,y,x])
    end
  end

  nothing
end

# """
# FFT of ETPC, i.e. P(ky, kx)
# """
# etpc_k!(args...) = begin etpc!(args...); rfft(mc.s.meas.etpc, (1,2)); end

# """
# P = Σ_r P(r), with P(r) from `etpc`.
# """
# etpc_uniform!(mc, η, greens) = begin etpc!(mc, η, greens); sum(mc.s.meas.etpc); end

# """
# P = P(q=0), with P(q)=fft(P(r)) from `etpc_k`. Slower than `etpc_uniform!`.
# """
# etpc_uniform_alternative!(mc, η, greens) = real(etpc_k!(mc, η, greens)[1,1])

"""
Equal time pairing correlation expectation value

  etpc_evs(j,y,x) = 1/N Σ_r0 << c_j1(r + r0) c_j2(r + r0) c_j3(r0)^† c_j4(r0)^† >>

where j ∈ 1:4 corresponds to (j1, j2, j3, j4) ∈ ((xd,xu,xu,xd), (xd,xu,yu,yd), (yd,yu,xu,xd), (yd,yu,yu,yd)),
r=(x,y), and we mean over r_0.
"""
function etpc_evs!(mc::AbstractDQMC, greens::AbstractMatrix)
  L = mc.p.L
  N = mc.l.sites
  etpc_ev = mc.s.meas.etpc_evs

  # we only need four combinations of j1,j2,j3,j4 for etpc_ev
  xu, yd, xd, yu = 1,2,3,4
  js = ((xd,xu,xu,xd), (xd,xu,yu,yd), (yd,yu,xu,xd), (yd,yu,yu,yd))

  fill!(etpc_ev, 0.)
  sql = reshape(collect(1:N),L,L)

  @inbounds for x in 0:(L-1), y in 0:(L-1) # r (displacement)
    for x0 in 1:L, y0 in 1:L # r0 (origin)
      r1 = r2 = siteidx(mc, sql, x0+x, y0+y)
      r3 = r4 = siteidx(mc, sql, x0, y0)

      for j in 1:length(js) # @inbounds
        i1 = greensidx(N, js[j][1], r1)
        i2 = greensidx(N, js[j][2], r2)
        i3 = greensidx(N, js[j][3], r3)
        i4 = greensidx(N, js[j][4], r4)

        # Wick's theorem. Result is real because greens is symm. in r and has anti-unitary symm.
        etpc_ev[j, y+1, x+1] += real(G(mc, i1, i4, greens)*G(mc, i2, i3, greens) -
                                     G(mc, i1, i3, greens)*G(mc, i2, i4, greens))
      end
    end
  end

  etpc_ev ./= N # r0 mean

  nothing
end



"""
Get linear site index of cartesian coordinates (x,y) respecting PBC.
"""
@inline siteidx(mc,sql,x,y) = siteidx(mc.p.L, mc.l.sites, sql, x, y)
@inline siteidx(L,N,sql,x,y) = begin
  xpbc = mod1(x, L)
  ypbc = mod1(y, L)
  sql[ypbc, xpbc] # to linear idx
end

"""
Get column/row idx of particular flv ∈ (xu, yd, xd, yu) and site ∈ 1:N in Green's function.
"""
@inline greensidx(N::Int, flv, site) = (flv-1)*N + site

"""
Access full (4*N, 4*N) Green's function for any OPDIM efficiently.
"""
@inline function G(mc, i, j, greens=mc.s.greens)
  gt = eltype(greens)
  N = mc.l.sites
  opdim = mc.p.opdim
  half = 2*N

  @inbounds if opdim == 3
    return greens[i,j]

  else
    # Virtually expand Green's function
    if i>half && j<=half # lower left block
      return zero(gt)

    elseif i<=half && j>half # upper right block
      return zero(gt)

    elseif i>half && j>half # lower right block
      return conj(greens[j - half, i - half])

    else # upper left block
      return greens[i, j]
    end
  end
end

"""
Construct full (4*N, 4*N) Green's function for any OPDIM efficiently.
"""
function fullG(mc, greens=mc.s.greens)
  gt = eltype(greens)
  N = mc.l.sites
  g = zeros(gt, 4*N, 4*N)

  @inbounds for j in 1:4*N
    for i in 1:4*N
      g[i,j] = G(mc, i, j, greens)
    end
  end
  return g
end






# -------------------------------------------------------
#         Postprocessing/Analysis
# -------------------------------------------------------
# Go from xup, ydown, xdown, yup -> xup, yup, xdown, ydown
function permute_greens(greens::AbstractMatrix)
  perm = [1,4,3,2] # flv*spin: xup, ydown, xdown, yup -> xup, yup, xdown, ydown
  N = Int(sqrt(length(greens))/4)
  return reshape(reshape(greens, (N,4,N,4))[:,perm,:,perm], (4*N,4*N)); # rfs1, rfs2
end

# Assuming translational invariance go to momentum space Greens function (k, fs1, fs2)
try
  eval(Expr(:toplevel, Expr(:using, Symbol("PyCall"))))
  eval(Expr(:toplevel, Meta.parse("@pyimport numpy as np")))
catch
end
function fft_greens(greens::AbstractMatrix)
  if !isdefined(:PyCall)
    eval(Expr(:toplevel, Expr(:using, Symbol("PyCall"))))
    eval(Expr(:toplevel, Meta.parse("@pyimport numpy as np")))
    println("Loaded PyCall. Please execute again.")
    return
  end

  L = Int(sqrt(sqrt(length(greens))/4))
  g = reshape(greens, (L,L,4,L,L,4)); # y1, x1, fs1, y2, x2, fs2
  g = fft(g, (1,2))*1/L; # ky1, kx1, fs1, y2, x2, fs2
  g = fft(g, (4,5))*1/L; # ky1, kx1, fs1, ky2, kx2, fs2
  g = reshape(g, (L*L,4,L*L,4))

  # check translational invariance
  println("translational invariance / momentum off diagonal terms")
  for jk1 in 1:4
      for jk2 in 1:4
          if jk1 !=jk2
              error1 = maximum(abs.(g[jk1,:,jk2,:]))
              error2 = maximum(abs.(g[jk2,:,jk1,:]))

              @printf("%d %d %.5e %.5e\n", jk1, jk2, error1, error2)
          end
      end
  end

  return np.einsum("kmkn->kmn", g); #k, fs1, fs2
end



# -------------------------------------------------------
#         Equal times Green's function (effective)
# -------------------------------------------------------
"""
QR DECOMPOSITION: Calculate effective(!) Green's function (direct, i.e. without stack)
"""
function calc_Bchain(mc::AbstractDQMC, start::Int, stop::Int, safe_mult::Int=mc.p.safe_mult)
  # Calculate Ul, Dl, Tl =B(stop) ... B(start)
  flv = mc.p.flv
  N = mc.l.sites
  G = geltype(mc)

  @assert 0 < start <= mc.p.slices
  @assert 0 < stop <= mc.p.slices
  @assert start <= stop

  U = Matrix{G}(I, flv*N, flv*N)
  D = ones(Float64, flv*N)
  T = Matrix{G}(I, flv*N, flv*N)

  svs = zeros(flv*N,length(start:stop))
  svc = 1
  for k in start:stop
    if mod(k,safe_mult) == 0 || k == stop # always decompose in the end
      multiply_B_left!(mc,k,U)
      rmul!(U, Diagonal(D))
      U, Tnew = decompose_udt!(U, D)
      mul!(mc.s.tmp, Tnew, T)
      T .=  mc.s.tmp
      svs[:,svc] = log.(D)
      svc += 1
    else
      multiply_B_left!(mc,k,U)
    end
  end
  return (U,D,T,svs)
end

# Calculate Ul, Dl, Tl = [B(stop) ... B(start)]^(-1) = B(start)^(-1) ... B(stop)^(-1)
function calc_Bchain_inv(mc::AbstractDQMC, start::Int, stop::Int, safe_mult::Int=mc.p.safe_mult)
  flv = mc.p.flv
  N = mc.l.sites
  G = geltype(mc)

  @assert 0 < start <= mc.p.slices
  @assert 0 < stop <= mc.p.slices
  @assert start <= stop

  U = Matrix{G}(I, flv*N, flv*N)
  D = ones(Float64, flv*N)
  T = Matrix{G}(I, flv*N, flv*N)

  svs = zeros(flv*N,length(start:stop))
  svc = 1
  for k in reverse(start:stop)
    if mod(k,safe_mult) == 0 || k == start # always decompose in the end
      multiply_B_inv_left!(mc,k,U)
      rmul!(U, Diagonal(D))
      U, Tnew = decompose_udt!(U, D)
      mul!(mc.s.tmp, Tnew, T)
      T .=  mc.s.tmp
      svs[:,svc] = log.(D)
      svc += 1
    else
      multiply_B_inv_left!(mc,k,U)
    end
  end
  return (U,D,T,svs)
end

# Calculate (Ur, Dr, Tr)' = B(stop) ... B(start)  => Ur,Dr, Tr = B(start)' ... B(stop)'
function calc_Bchain_dagger(mc::AbstractDQMC, start::Int, stop::Int, safe_mult::Int=mc.p.safe_mult)
  flv = mc.p.flv
  N = mc.l.sites
  G = geltype(mc)

  @assert 0 < start <= mc.p.slices
  @assert 0 < stop <= mc.p.slices
  @assert start <= stop

  U = Matrix{G}(I, flv*N, flv*N)
  D = ones(Float64, flv*N)
  T = Matrix{G}(I, flv*N, flv*N)

  svs = zeros(flv*N,length(start:stop))
  svc = 1
  for k in reverse(start:stop)
    if mod(k,safe_mult) == 0 || k == start # always decompose in the end
      multiply_daggered_B_left!(mc,k,U)
      rmul!(U, Diagonal(D))
      U, Tnew = decompose_udt!(U, D)
      mul!(mc.s.tmp, Tnew, T)
      T .=  mc.s.tmp
      svs[:,svc] = log.(D)
      svc += 1
    else
      multiply_daggered_B_left!(mc,k,U)
    end
  end
  return (U,D,T,svs)
end

# Calculate G(slice) = [1+B(slice-1)...B(1)B(M) ... B(slice)]^(-1) and its singular values in a stable manner
function calc_greens(mc::AbstractDQMC, slice::Int=mc.s.current_slice, safe_mult::Int=mc.p.safe_mult)
  s = mc.s
  calc_greens_helper(mc, slice, safe_mult)

  lmul!(Diagonal(s.d), s.U)
  return s.T * s.U
end
function calc_greens_and_logdet(mc::AbstractDQMC, slice::Int=mc.s.current_slice, safe_mult::Int=mc.p.safe_mult)
  s = mc.s
  calc_greens_helper(mc, slice, safe_mult)

  ldet = real(log(complex(det(s.U))) + sum(log.(s.d)) + log(complex(det(s.T))))

  lmul!(Diagonal(s.d), s.U)
  return s.T * s.U, ldet
end
function calc_greens_udt(mc::AbstractDQMC, slice::Int, safe_mult::Int=mc.p.safe_mult)
  s = mc.s
  calc_greens_helper(mc, slice, safe_mult)

  # greens = s.T * Diagonal(s.d) * s.U
  lmul!(Diagonal(s.d), s.U)
  U, T = decompose_udt!(s.U, s.d)
  return s.T*U, copy(s.d), T
end

# result in s.T, s.U and s.d
function calc_greens_helper(mc::AbstractDQMC, slice::Int, safe_mult::Int)
  flv = mc.p.flv
  N = mc.l.sites
  G = geltype(mc)

  # Calculate Ur,Dr,Tr=B(slice)' ... B(M)'
  Ur, Dr, Tr = calc_Bchain_dagger(mc,slice,mc.p.slices, safe_mult)

  # Calculate Ul,Dl,Tl=B(slice-1) ... B(1)
  if slice-1 >= 1
    Ul, Dl, Tl = calc_Bchain(mc,1,slice-1, safe_mult)
  else
    Ul = Matrix{G}(I, flv*N, flv*N)
    Dl = ones(Float64, flv*N)
    Tl = Matrix{G}(I, flv*N, flv*N)
  end

  # calculate greens
  s = mc.s
  tmp = mc.s.tmp
  tmp2 = mc.s.tmp2

  mul!(tmp, Tl, adjoint(Tr))
  rmul!(tmp, Diagonal(Dr))
  lmul!(Diagonal(Dl), tmp)
  s.U, s.T = decompose_udt!(tmp, s.D)

  mul!(tmp, Ul, s.U)
  s.U .= tmp
  mul!(tmp2, s.T, adjoint(Ur))
  s.T .= tmp2
  mul!(tmp, adjoint(s.U), inv(s.T))
  tmp[diagind(tmp)] .+= s.D
  u, t = decompose_udt!(tmp, s.d)

  mul!(tmp, t, s.T)
  s.T = inv(tmp)
  mul!(tmp, s.U, u)
  s.U = adjoint(tmp)
  s.d .= 1 ./ s.d

  nothing
end



"""
SVD DECOMPOSITION: Calculate effective(!) Green's function (direct, i.e. without stack)
"""
function calc_Bchain_udv(mc::AbstractDQMC, start::Int, stop::Int, safe_mult::Int=mc.p.safe_mult)
  # Calculate B(stop) ... B(start) safely (with stabilization at every safe_mult step, default ALWAYS)
  # Returns: tuple of results (U, D, and V) and log singular values of the intermediate products
  flv = mc.p.flv
  slices = mc.p.slices
  N = mc.l.sites
  G = geltype(mc)

  @assert 0 < start <= slices
  @assert 0 < stop <= slices
  @assert start <= stop

  U = Matrix{G}(I, flv*N, flv*N)
  D = ones(Float64, flv*N)
  Vt = Matrix{G}(I, flv*N, flv*N)
  Vtnew = Matrix{G}(I, flv*N, flv*N)

  svs = zeros(flv*N,length(start:stop))
  svc = 1
  for k in start:stop
    if mod(k,safe_mult) == 0
      multiply_B_left!(mc,k,U)
      U *= Diagonal(D)
      U, D, Vtnew = decompose_udv!(U)
      # not yet in-place
      Vt =  Vtnew * Vt
      svs[:,svc] = log.(D)
      svc += 1
    else
      multiply_B_left!(mc,k,U)
    end
  end
  return (U,D,Vt,svs)
end

# Calculate G(slice) = [1+B(slice-1)...B(1)B(M) ... B(slice)]^(-1) and its logdet in a stable manner
function calc_greens_and_logdet_udv(mc::AbstractDQMC, slice::Int, safe_mult::Int=mc.p.safe_mult)
  flv = mc.p.flv
  N = mc.l.sites
  G = geltype(mc)

  # Calculate Ur,Dr,Vtr=B(M) ... B(slice)
  Ur, Dr, Vtr = calc_Bchain_udv(mc,slice,mc.p.slices,safe_mult)

  # Calculate Ul,Dl,Vtl=B(slice-1) ... B(1)
  if slice-1 >= 1
    Ul, Dl, Vtl = calc_Bchain_udv(mc,1,slice-1,safe_mult)
  else
    Ul = Matrix{G}(I, flv*N, flv*N)
    Dl = ones(Float64, flv*N)
    Vtl = Matrix{G}(I, flv*N, flv*N)
  end

  # Calculate Greens function
  tmp = Vtl * Ur
  inner = adjoint(Vtr * Ul) + Diagonal(Dl) * tmp * Diagonal(Dr)
  I = decompose_udv!(inner)
  U = adjoint(I[3] * Vtr)
  D = Diagonal(1 ./ I[2])
  Vt = adjoint(Ul * I[1])
  return U*D*Vt, sum(log.(diag(D)))
end

# -------------------------------------------------------
#    Effective Green's function -> Green's function
# -------------------------------------------------------
function effective_greens2greens!(mc::DQMC_CBTrue, greens::AbstractMatrix)
  chkr_hop_half_minus = mc.l.chkr_hop_half
  chkr_hop_half_plus = mc.l.chkr_hop_half_inv
  n_groups = mc.l.n_groups
  tmp = mc.s.tmp

  @inbounds @views begin
      for i in reverse(1:n_groups)
        mul!(tmp, greens, chkr_hop_half_minus[i])
        greens .= tmp
      end
      for i in reverse(1:n_groups)
        mul!(tmp, chkr_hop_half_plus[i], greens)
        greens .= tmp
      end
  end
  nothing
end
function effective_greens2greens!(mc::DQMC_CBTrue, U::AbstractMatrix, T::AbstractMatrix)
  chkr_hop_half_minus = mc.l.chkr_hop_half
  chkr_hop_half_plus = mc.l.chkr_hop_half_inv
  n_groups = mc.l.n_groups
  tmp = mc.s.tmp

  @inbounds @views begin
      for i in reverse(1:n_groups)
        mul!(tmp, T, chkr_hop_half_minus[i])
        T .= tmp
      end
      for i in reverse(1:n_groups)
        mul!(tmp, chkr_hop_half_plus[i], U)
        U .= tmp
      end
  end
  nothing
end
function greens2effective_greens!(mc::DQMC_CBTrue, greens::AbstractMatrix)
  chkr_hop_half_minus = mc.l.chkr_hop_half
  chkr_hop_half_plus = mc.l.chkr_hop_half_inv
  n_groups = mc.l.n_groups

  @inbounds @views begin
      for i in 1:n_groups
        mul!(tmp, greens, chkr_hop_half_plus[i])
        greens .= tmp
      end
      for i in 1:n_groups
        mul!(tmp, chkr_hop_half_minus[i], greens)
        greens .= tmp
      end
  end
  nothing
end
function effective_greens2greens!(mc::DQMC_CBFalse, greens::AbstractMatrix)
  eTminus = mc.l.hopping_matrix_exp
  eTplus = mc.l.hopping_matrix_exp_inv
  tmp = mc.s.tmp

  mul!(tmp, greens, eTminus)
  mul!(greens, eTplus, tmp)
  nothing
end
function effective_greens2greens!(mc::DQMC_CBFalse, U::AbstractMatrix, T::AbstractMatrix)
  eTminus = mc.l.hopping_matrix_exp
  eTplus = mc.l.hopping_matrix_exp_inv

  T .= T * eTminus
  U .= eTplus * U
  nothing
end
function greens2effective_greens!(mc::DQMC_CBFalse, greens::AbstractMatrix)
  eTminus = mc.l.hopping_matrix_exp
  eTplus = mc.l.hopping_matrix_exp_inv
  tmp = mc.s.tmp

  mul!(tmp, greens, eTplus)
  mul!(greens, eTminus, tmp)
  nothing
end
function effective_greens2greens(mc::AbstractDQMC, greens::AbstractMatrix)
  g = copy(greens)
  effective_greens2greens!(mc, g)
  return g
end



# -------------------------------------------------------
#         Time-displaced Green's function
# -------------------------------------------------------
# Calculate "G(tau, 0)", i.e. G(slice,1) as G(slice,1) = B(slice,1)G(1)
# function calc_tdgf_naive(mc::AbstractDQMC, slice::Int, safe_mult::Int=mc.p.safe_mult)
#   U,D,T = calc_greens_udt(mc, 1, safe_mult)

#   # effective -> actual
#   effective_greens2greens!(mc, U, T)

#   # time displace
#   Ul, Dl, Tl = calc_Bchain(mc, 1, slice, safe_mult)
#   U, D, T = multiply_safely(Ul, Dl, Tl, U, D, T)

#   rmul!(U, Diagonal(D))
#   return U*T
# end

# Calculate "G(tau, 0)", i.e. G(slice,1) as G(slice,1) = [B(slice, 1)^-1 + B(beta, slice)]^-1 which is equal to B(slice,1)G(1)
function calc_tdgf(mc::AbstractDQMC, slice::Int, safe_mult::Int=mc.p.safe_mult)
  if slice != 1
    Ul, Dl, Tl = calc_Bchain_inv(mc, 1, slice-1, safe_mult)
  else
    Ul, Dl, Tl = mc.s.eye_full, mc.s.ones_vec, mc.s.eye_full
  end

  if slice != mc.p.slices
    Ur, Dr, Tr = calc_Bchain(mc, slice, mc.p.slices, safe_mult)
  else
    Ur, Dr, Tr = mc.s.eye_full, mc.s.ones_vec, mc.s.eye_full
  end

  # time displace
  U, D, T = inv_sum_udts(Ul, Dl, Tl, Ur, Dr, Tr)
  effective_greens2greens!(mc, U, T)

  rmul!(U, Diagonal(D))
  return U*T
end









global LEFT = true
global RIGHT = false
"""
Calculate UDVs at safe_mult time slices of
dir = LEFT: 
inv=false:  B(tau, 1) = B(tau) * B(tau-1) * ... * B(1)                    # mult left, 1:tau
inv=true:   [B(tau, 1)]^-1 = B(1)^-1 * B(2)^-1 * ... B(tau)^-1            # mult inv right, 1:tau

udv[i] = from 1 to mc.s.ranges[i][end]


dir = RIGHT:
inv=false:  B(beta, tau) = B(beta) * B(beta-1) * ... * B(tau)             # mult right, beta:tau
inv=true:   [B(beta, tau)]^-1 = B(tau)^-1 * B(tau+1)^-1 * ... B(beta)^-1  # mult inv left, beta:tau

udv[i] = from mc.s.ranges[i][1] to mc.p.slices (beta)
"""
function calc_tdgf_B_udvs(mc::AbstractDQMC; inv::Bool=false, dir::Bool=LEFT)
  G = geltype(mc)
  flv = mc.p.flv
  N = mc.l.sites
  nranges= length(mc.s.ranges)
  curr_U_or_T = mc.s.curr_U
  eye_full = mc.s.eye_full
  ones_vec = mc.s.ones_vec
  ranges = mc.s.ranges
  
  u_stack = [zeros(G, flv*N, flv*N) for _ in 1:nranges]
  d_stack = [zeros(Float64, flv*N) for _ in 1:nranges]
  t_stack = [zeros(G, flv*N, flv*N) for _ in 1:nranges]

  rightmult = false
  ((dir == RIGHT && !inv) || (dir == LEFT && inv)) && (rightmult = true)

  # @show rightmult

  rng = 1:length(ranges)
  dir == RIGHT && (rng = reverse(rng))

  # Calculate udv[i], given udv[i-1]
  @inbounds for (i, rngidx) in enumerate(rng)

    if i == 1
      copyto!(curr_U_or_T, eye_full)
    else
      if !rightmult
        copyto!(curr_U_or_T, u_stack[i-1])
      else
        copyto!(curr_U_or_T, t_stack[i-1])
      end
    end

    slice_range = dir == RIGHT ? reverse(ranges[rngidx]) : ranges[rngidx]

    for slice in slice_range
      if inv == false
        if dir == LEFT
          multiply_B_left!(mc, slice, curr_U_or_T)
        else
          # rightmult
          multiply_B_right!(mc, slice, curr_U_or_T)
        end
      else
        if dir == LEFT
          # rightmult
          multiply_B_inv_right!(mc, slice, curr_U_or_T)
        else
          multiply_B_inv_left!(mc, slice, curr_U_or_T)
        end
      end
    end

    if i != 1
      if !rightmult
        rmul!(curr_U_or_T, Diagonal(d_stack[i-1]))
      else
        lmul!(Diagonal(d_stack[i-1]), curr_U_or_T)
      end
    end

    if !rightmult
      u_stack[i], T = decompose_udt!(curr_U_or_T, d_stack[i])
    else
      U, t_stack[i] = decompose_udt!(curr_U_or_T, d_stack[i])
    end

    if i == 1
      if !rightmult
        mul!(t_stack[i], T, eye_full)
      else
        mul!(u_stack[i], eye_full, U)
      end
    else
      if !rightmult
        mul!(t_stack[i], T, t_stack[i-1])
      else
        mul!(u_stack[i], u_stack[i-1], U)
      end
    end
  end

  if dir == LEFT
    return u_stack, d_stack, t_stack
  else
    return reverse(u_stack), reverse(d_stack), reverse(t_stack)
  end
end


function calc_tdgfs!(mc)
  G = geltype(mc)
  M = mc.p.slices
  N = mc.l.sites
  flv = mc.p.flv
  Nflv = N * flv
  safe_mult = mc.p.safe_mult
  eye_full = mc.s.eye_full
  ones_vec = mc.s.ones_vec

  # allocate matrices if not yet done
  try
    mc.s.Gt0[1]
    mc.s.G0t[1]
  catch
    mc.s.Gt0 = [zeros(G, Nflv, Nflv) for _ in 1:M]
    mc.s.G0t = [zeros(G, Nflv, Nflv) for _ in 1:M]
  end

  Gt0 = mc.s.Gt0
  G0t = mc.s.G0t

  # ---- first, calculate Gt0 and G0t only at safe_mult slices 
  # right mult (Gt0)
  BT0Inv_u_stack, BT0Inv_d_stack, BT0Inv_t_stack = calc_tdgf_B_udvs(mc, inv=true, dir=LEFT);
  BBetaT_u_stack, BBetaT_d_stack, BBetaT_t_stack = calc_tdgf_B_udvs(mc, inv=false, dir=RIGHT);
  
  # left mult (G0t)
  BT0_u_stack, BT0_d_stack, BT0_t_stack = calc_tdgf_B_udvs(mc, inv=false, dir=LEFT);
  BBetaTInv_u_stack, BBetaTInv_d_stack, BBetaTInv_t_stack = calc_tdgf_B_udvs(mc, inv=true, dir=RIGHT);



  safe_mult_taus = 1:safe_mult:mc.p.slices
  @inbounds for i in 1:length(safe_mult_taus) # i = ith safe mult time slice
    tau = safe_mult_taus[i] # tau = tauth (overall) time slice
    if i != 1
      # Gt0
      inv_sum_udts!(mc, Gt0[tau], BT0Inv_u_stack[i-1], BT0Inv_d_stack[i-1], BT0Inv_t_stack[i-1],
                   BBetaT_u_stack[i], BBetaT_d_stack[i], BBetaT_t_stack[i]) # G(i,0) = G(mc.s.ranges[i][1], 0), i.e. G(21, 1) for i = 3
      effective_greens2greens!(mc, Gt0[tau])

      # G0t
      inv_sum_udts!(mc, G0t[tau], BT0_u_stack[i-1], BT0_d_stack[i-1], BT0_t_stack[i-1],
                   BBetaTInv_u_stack[i], BBetaTInv_d_stack[i], BBetaTInv_t_stack[i]) # G(i,0) = G(mc.s.ranges[i][1], 0), i.e. G(21, 1) for i = 3
      effective_greens2greens!(mc, G0t[tau])
    else
      # Gt0
      inv_one_plus_udt!(mc, Gt0[tau], BBetaT_u_stack[1], BBetaT_d_stack[1], BBetaT_t_stack[1])
      effective_greens2greens!(mc, Gt0[tau])

      # G0t
      inv_one_plus_udt!(mc, G0t[tau], BBetaTInv_u_stack[1], BBetaTInv_d_stack[1], BBetaTInv_t_stack[1])
      effective_greens2greens!(mc, G0t[tau]) # TODO: check analytically that we can still do this
    end
  end

  # ---- fill time slices between safe_mult slices
  fill_tdgf!(mc, Gt0, G0t)

  @inbounds for i in 1:M
    G0t[i] .*= -1 # minus sign
  end

  nothing
end


# Given Gt0 and G0t at safe mult slices (mc.s.ranges[i][1])
# propagate to all other slices.
function fill_tdgf!(mc, Gt0, G0t)
  safe_mult = mc.p.safe_mult
  M = mc.p.slices

  safe_mult_taus = 1:safe_mult:M
  @inbounds for tau in 1:M
    (tau in safe_mult_taus) && continue # skip safe mult taus

    Gt0[tau] .= Gt0[tau-1] # copy
    multiply_B_left!(mc, tau, Gt0[tau])

    G0t[tau] .= G0t[tau-1] # copy
    multiply_B_inv_right!(mc, tau, G0t[tau])
  end

  nothing
end





function test_Gt0()
  eye_full = mc.s.eye_full
  ones_vec = mc.s.ones_vec

  Gt0 = zeros(G, Nflv, Nflv)

  # i = 3
  # i = 11 # == 101, almost beta half = 100
  i = 1

  if i != 1
    U,D,T = inv_sum_udts(BT0Inv_u_stack[i-1], BT0Inv_d_stack[i-1], BT0Inv_t_stack[i-1],
                 BBetaT_u_stack[i], BBetaT_d_stack[i], BBetaT_t_stack[i])
    UDT_to_mat!(Gt0, U, D, T) # G(i,0) = G(mc.s.ranges[i][1], 0), i.e. G(21, 1) for i = 3
    effective_greens2greens!(mc, Gt0)
  else
    U,D,T = inv_sum_udts(eye_full, ones_vec, eye_full,
                 BBetaT_u_stack[i], BBetaT_d_stack[i], BBetaT_t_stack[i])
    UDT_to_mat!(Gt0, U, D, T) # G(i,0) = G(mc.s.ranges[i][1], 0), i.e. G(21, 1) for i = 3
    effective_greens2greens!(mc, Gt0)
  end

  tdgf = calc_tdgf(mc, mc.s.ranges[i][1]);
  compare(tdgf, Gt0)

  # compare G(0,0) with G(0) for i=1
  g = calc_greens(mc, 1);
  compare(Gt0, g) # 1e-16 for i=1


  # test all Gt0 at safe mult slices
  safe_mult_taus = 1:safe_mult:mc.p.slices
  for tau in safe_mult_taus
    i = ceil(Int, tau/safe_mult)
    tdgf = calc_tdgf(mc, mc.s.ranges[i][1]);
    if !isapprox(tdgf, Gt0[tau])
      @show tau
      @show i
      break
    end
  end
  # worked!

  # TODO: test all G0t at safe mult slices
end





function test_stacks()
  nr = length(mc.s.ranges)

  check_unitarity(BT0_u_stack)
  check_unitarity(BBetaTInv_u_stack)
  check_unitarity(BT0Inv_u_stack)
  check_unitarity(BBetaT_u_stack)



  # test left multiplications
  B2 = BT0_u_stack[3] * Diagonal(BT0_d_stack[3]) * BT0_t_stack[3];
  U, D, T = calc_Bchain(mc, 1, mc.s.ranges[3][end]); B1 = U*Diagonal(D)*T;
  compare(B1, B2) # this is exactly the same

  B2 = BBetaTInv_u_stack[3] * Diagonal(BBetaTInv_d_stack[3]) * BBetaTInv_t_stack[3];
  U, D, T = calc_Bchain_inv(mc, mc.s.ranges[3][1], mc.p.slices); B1 = U*Diagonal(D)*T;
  compare(B1, B2) # why is there a difference here at all?


  # test right multiplications
  B2 = BT0Inv_u_stack[3] * Diagonal(BT0Inv_d_stack[3]) * BT0Inv_t_stack[3];
  U, D, T = calc_Bchain_inv(mc, 1, mc.s.ranges[3][end]); B1 = U*Diagonal(D)*T;
  compare(B1, B2)

  B2 = BBetaT_u_stack[3] * Diagonal(BBetaT_d_stack[3]) * BBetaT_t_stack[3];
  U, D, T = calc_Bchain(mc, mc.s.ranges[3][1], mc.p.slices); B1 = U*Diagonal(D)*T;
  compare(B1, B2)






  # compare B(beta,1) from BT0 and BBetaT
  BT0_full = BT0_u_stack[end] * Diagonal(BT0_d_stack[end]) * BT0_t_stack[end];
  BBetaT_full = BBetaT_u_stack[1] * Diagonal(BBetaT_d_stack[1]) * BBetaT_t_stack[1];
  U, D, T = calc_Bchain(mc, 1, mc.s.ranges[end][end]); BBeta0 = U*Diagonal(D)*T;
  compare(BT0_full, BBeta0)
  compare(BBetaT_full, BBeta0) # we have (large) abs errors here. maybe it's still ok

  # compare resulting greens
  gT0_full = inv_one_plus_udt(BT0_u_stack[end], BT0_d_stack[end], BT0_t_stack[end])
  gBetaT_full = inv_one_plus_udt(BBetaT_u_stack[1], BBetaT_d_stack[1], BBetaT_t_stack[1])
  gBeta0 = calc_greens(mc, 1)
  compare(gT0_full, gBeta0) # 1e-16
  compare(gBetaT_full, gBeta0) # 1e-16






  # compare B(beta, 1), build by combining BBetaT and BT0
  i = floor(Int, nr/2)

  U = BBetaT_t_stack[i+1] * BT0_u_stack[i]
  rmul!(U, Diagonal(BT0_d_stack[i]))
  lmul!(Diagonal(BBetaT_d_stack[i+1]), U)
  u,d,t = decompose_udt(U)
  u = BBetaT_u_stack[i+1] * u
  t = t * BT0_t_stack[i]
  Bfull = u * Diagonal(d) * t
  Bfull_d = copy(d)
  U, D, T = calc_Bchain(mc, 1, mc.p.slices); Bfull2 = U * Diagonal(D) * T;
  Bfull2_d = copy(D)
  compare(Bfull, Bfull2) # max absdiff: 7.7e+03, max reldiff: 3.4e+01
  compare(Bfull_d, Bfull2_d) # max absdiff: 9.2e+03, max reldiff: 2.3e-13

  # compare resulting greens
  g1 = inv_one_plus_udt(u, d, t)
  g2 = calc_greens(mc, 1)
  compare(g1, g2) # 1e-15
end

function check_unitarity(u_stack)
  for i in 1:length(u_stack)
    U = u_stack[i]
    !isapprox(U * adjoint(U), I) && (return false) # I was eye(U)
  end
  return true
end



# -------------------------------------------------------
#                Correlation functions
# -------------------------------------------------------

# function tdpc()

#   for t in 1:slices
#     for Δx in 0:L-1, Δy in 0:L-1
#       for x in 1:L, y in 1:L
#         j1 = linidx(x,y)
#         j2 = linidx(x,y)
#         j3 = linidx(x+Δx, y+Δy)
#         j4 = linidx(x+Δx, y+Δy)

#         if p.op_dim == 3
#           g14 = tdgf[t]...
#           g23 = tdgf[t]...
#           g13 = tdgf[t]...
#           g24 = tdgf[t]...
#         else
#           # lazily expand to full GF
#         end

#         for 
#       end
#     end
#   end

# end


# Yoni



# function inv_sum(U1,D1,T1,U2,D2,T2)
#   m1 = T1 * inv(T2)
#   lmul!(Diagonal(D1), m1)
#   m2 = adjoint(U1) * U2
#   rmul!(m2, Diagonal(D2))

#   u,d,t = decompose_udt(m1+m2)

#   A = inv(t*T2)
#   B = 1 ./ d
#   C = adjoint(U1*u)

#   lmul!(Diagonal(B), C)
#   return A*C
# end