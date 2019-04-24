import DSP.fftfreq

# -------------------------------------------------------
#           Green's function indexing stuff
# -------------------------------------------------------
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
      return conj(greens[i - half, j - half])

    else # upper left block
      return greens[i, j]
    end
  end
end

"""
Access full (4*N, 4*N) `Gtilde = I - G` where G is the Green's function for any OPDIM efficiently.
"""
@inline function Gtilde(mc, i, j, greens=mc.s.greens)
  if i == j # on diagonal
    return 1 - G(mc, i, j, greens)
  else
    return -G(mc, i, j, greens)
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

"""
Construct full (4*N, 4*N) `Gtilde = I - G` where G is the Green's function for any OPDIM efficiently.
"""
function fullGtilde(mc, greens=mc.s.greens)
  gt = eltype(greens)
  N = mc.l.sites
  g = zeros(gt, 4*N, 4*N)

  @inbounds for j in 1:4*N
    for i in 1:4*N
      g[i,j] = Gtilde(mc, i, j, greens)
    end
  end
  return g
end

# test: fullGtilde(mc) == (I - fullG(mc))







# -------------------------------------------------------
#         Postprocessing/Analysis
# -------------------------------------------------------
# Go from xup, ydown, xdown, yup -> xup, yup, xdown, ydown
function permute_greens(greens::AbstractMatrix)
  perm = [1,4,3,2] # flv*spin: xup, ydown, xdown, yup -> xup, yup, xdown, ydown
  N = Int(sqrt(length(greens))/4)
  return reshape(reshape(greens, (N,4,N,4))[:,perm,:,perm], (4*N,4*N)); # rfs1, rfs2
end

"""
    fft_greens(greens; fftshift=false, flv=4, L=sqrt(size(greens, 1)/flv))

Fourier transform real-space Green's function (`flv*N, flv*N`) to momentum space.

If `fftshift == true` momenta are shifted such that k=0 (Γ point) is centered.
"""
function fft_greens(greens::AbstractMatrix; fftshift=false, flv::Integer=4, L::Integer=Int(sqrt(size(greens, 1)/flv)))
  N = L^2
  g = reshape(greens, (L,L,flv,L,L,flv)) # y1,x1,f1, y2,x2,f2
  gk = ifft( fft(g, (1,2)), (4,5))
  if fftshift
    gk = FFTW.fftshift(gk, (1,2,4,5))
  end
  gk = reshape(gk, (flv*N,flv*N))
end

fft_greens(mc::AbstractDQMC, greens::AbstractMatrix; fftshift=false) = fft_greens(greens; fftshift=fftshift, flv=mc.p.flv, L=mc.p.L)


"""
    ifft_greens(gk; ifftshift=false, flv=4, L=sqrt(size(greens, 1)/flv))

Inverse Fourier transform momentum-space Green's function (`flv*N, flv*N`) to real space.
Assumes fftshifted momenta as it applies ifftshift before the transformation.

Inverse of `fft_greens`.
"""
function ifft_greens(gk::AbstractMatrix; ifftshift=false, flv::Integer=4, L::Integer=Int(sqrt(size(gk, 1)/flv)))
  N = L^2
  g = reshape(gk, (L,L,flv,L,L,flv)) # y1,x1,f1, y2,x2,f2
  if ifftshift
    g = FFTW.ifftshift(g, (1,2,4,5))
  end
  greens = fft( ifft(g, (1,2)), (4,5))
  greens = reshape(greens, (flv*N,flv*N))
end

ifft_greens(mc::AbstractDQMC, gk::AbstractMatrix; ifftshift=false) = ifft_greens(gk; ifftshift=ifftshift, flv=mc.p.flv, L=mc.p.L)




"""
    fftmomenta(L; fftshift=false)

Returns momenta of a finite size system (Brillouin zone discretization).

If `fftshift==true` the momenta are shifted such that k=0 (Γ point) is centered.

The momenta are those of the output of `fft_greens`.
"""
function fftmomenta(L::Int; fftshift=false)
  qs = 2*pi*fftfreq(L) # 2*pi because fftfreq returns linear momenta
  if fftshift
    qs = FFTW.fftshift(qs)
  end

  # ωs = 2*pi*fftfreq(M, 1/Δτ)
  return qs
end


fftmomenta(mc::AbstractDQMC; kwargs...) = fftmomenta(mc.l.sites; kwargs...)








# -------------------------------------------------------
#             Equal times Green's function
# -------------------------------------------------------
function measure_greens_and_logdet(mc::AbstractDQMC, safe_mult::Int=mc.p.safe_mult, slice::Int=1)
  greens, greens_logdet = calc_greens_and_logdet(mc, slice, safe_mult)
  effective_greens2greens!(mc, greens)
  return greens, greens_logdet
end

function measure_greens(mc::AbstractDQMC, safe_mult::Int=mc.p.safe_mult, slice::Int=1)
  greens = calc_greens(mc, slice, safe_mult)
  effective_greens2greens!(mc, greens)
  return greens
end







# -------------------------------------------------------
#   Equal times Green's function (all of them, i.e. ∀τ)
# -------------------------------------------------------
"""
Calculates (true i.e. not effective) equal-time Green's functions
at every time slice, i.e. G(tau) for all tau in 1:M where M is # slices.
"""
function measure_all_greens(mc::AbstractDQMC)
  etgfs = calc_all_greens(mc)
  effective_greens2greens!.(Ref(mc), etgfs)
  return etgfs
end


"""
Calculates (effective) equal-time Green's functions at every
time slice, i.e. G(tau) for all tau in 1:M where M is # slices.

This method is using the dqmc stack and its logic.
"""
function calc_all_greens(mc::AbstractDQMC)
  build_stack(mc)
  M = mc.p.slices
  safe_mult = mc.p.safe_mult

  etgfs = Vector{Matrix{geltype(mc)}}(undef, M)

  for tau in M:-1:1
    propagate(mc)
    # @assert tau == mc.s.current_slice
    @inbounds etgfs[tau] = copy(mc.s.greens)
  end

  return etgfs
end


"""
Calculates (effective) equal-time Green's functions at every
time slice, i.e. G(tau) for all tau in 1:M where M is # slices.
"""
function calc_all_greens_explicit(mc::AbstractDQMC)
  # TODO: remove calculation redundancy by storing intermediate Bchains
  M = mc.p.slices
  safe_mult = mc.p.safe_mult

  etgfs = Vector{Matrix{geltype(mc)}}(undef, M)

  etgfs[1] = calc_greens(mc, 1)

  for tau in 2:M
    if mod(tau-1, safe_mult) == 0
      etgfs[tau] = calc_greens(mc, tau)
    else
      etgfs[tau] = wrap_greens(mc, etgfs[tau-1], tau-1, 1)
    end
  end

  return etgfs
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
#           Equal-time Pairing Correlations
# -------------------------------------------------------
function allocate_etpc!(mc)
  L = mc.p.L
  meas = mc.s.meas

  meas.etpc_minus = zeros(Float64,L,L)
  meas.etpc_plus = zeros(Float64,L,L)

  println("Allocated memory for ETPC measurements.")
  nothing
end


"""
Calculate equal time pairing susceptibilities (Eq. 4 in notes)

  Pη(y,x) = << Δη^†(r)Δη(0) >> = etpc_plus/minus(y,x)

where η=± (s and d-wave), r=(y,x), and <<·>> indicates a fermion average for fixed ϕ.

x,y ∈ [0,L-1]

Details:
- we mean over r_0, that is "0".
- we do not mean over τ_0=0 (τ not shown above).
"""
function etpc!(mc::AbstractDQMC, greens::AbstractMatrix)
  L = mc.p.L
  N = mc.l.sites
  Pm = mc.s.meas.etpc_minus # d-wave
  Pp = mc.s.meas.etpc_plus # s-wave

  fill!(Pm, 0.0)
  fill!(Pp, 0.0)

  # we only need four combinations of j1,j2,j3,j4 for etpc_ev
  xu, yd, xd, yu = 1,2,3,4
  js = ((xd,xu,xu,xd), (xd,xu,yu,yd), (yd,yu,xu,xd), (yd,yu,yu,yd))

  sql = reshape(1:N,L,L)


  @inbounds for x in 0:(L-1), y in 0:(L-1) # r (displacement)
    for x0 in 1:L, y0 in 1:L # r0 (origin)
      r1 = r2 = siteidx(mc, sql, x0+x, y0+y)
      r3 = r4 = siteidx(mc, sql, x0, y0)

      ev1 = _pc_ev(mc, greens, r1, r2, r3, r4, js, 1)
      ev2 = _pc_ev(mc, greens, r1, r2, r3, r4, js, 2)
      ev3 = _pc_ev(mc, greens, r1, r2, r3, r4, js, 3)
      ev4 = _pc_ev(mc, greens, r1, r2, r3, r4, js, 4)

      Pm[y+1,x+1] += ev1 - ev2 - ev3 + ev4
      Pp[y+1,x+1] += ev1 + ev2 + ev3 + ev4
    end
  end

  # r0 mean
  Pm ./= N
  Pp ./= N

  nothing
end



"""
Pairing correlation expectation value

  etpc_ev = << c_j1(r1) c_j2(r2) c_j3(r3)^† c_j4(r4)^† >>

(Wick's theorem.)
"""
@inline function _pc_ev(mc, greens, r1, r2, r3, r4, js, j)
  N = mc.l.sites

  i1 = greensidx(N, js[j][1], r1)
  i2 = greensidx(N, js[j][2], r2)
  i3 = greensidx(N, js[j][3], r3)
  i4 = greensidx(N, js[j][4], r4)

  # Wick's theorem. Result is real because greens is symm. in r and has anti-unitary symm.
  result = G(mc, i1, i4, greens)*G(mc, i2, i3, greens) -
              G(mc, i1, i3, greens)*G(mc, i2, i4, greens)
  # @show maximum(imag(result))
  # @assert isreal(result)
  return real(result)
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






# -------------------------------------------------------
#      Zero-Frequency (static) Pairing Correlations
# -------------------------------------------------------
function allocate_zfpc!(mc)
  L = mc.p.L
  meas = mc.s.meas

  meas.zfpc_minus = zeros(Float64,L,L)
  meas.zfpc_plus = zeros(Float64,L,L)

  println("Allocated memory for ZFPC measurements.")
  nothing
end


"""
Calculate zero-frequency pairing susceptibilities (Eq. 3 in notes)

  Pη(y,x) = int_tau << Δη^†(r, tau)Δη(0, 0) >> = zfpc_plus/minus(y,x)

where η=± (s and d-wave), r=(y,x), and <<·>> indicates a fermion average for fixed ϕ.

x,y ∈ [0,L-1]

Details:
- we mean over r_0, that is "0".
- we do not mean over τ_0=0.
"""
function zfpc!(mc::AbstractDQMC, Gt0s::AbstractVector{T}) where T <: AbstractMatrix
  L = mc.p.L
  N = mc.l.sites
  M = mc.p.slices
  Pm = mc.s.meas.zfpc_minus # d-wave
  Pp = mc.s.meas.zfpc_plus # s-wave

  fill!(Pm, 0.0)
  fill!(Pp, 0.0)

  # we only need four combinations of j1,j2,j3,j4 for etpc_ev
  xu, yd, xd, yu = 1,2,3,4
  js = ((xd,xu,xu,xd), (xd,xu,yu,yd), (yd,yu,xu,xd), (yd,yu,yu,yd))

  sql = reshape(1:N,L,L)

  @inbounds for tau in 1:M
    greens = Gt0s[tau]

    # Identical to etpc now
    @inbounds for x in 0:(L-1), y in 0:(L-1) # r (displacement)
      for x0 in 1:L, y0 in 1:L # r0 (origin)
        r1 = r2 = siteidx(mc, sql, x0+x, y0+y)
        r3 = r4 = siteidx(mc, sql, x0, y0)

        ev1 = _pc_ev(mc, greens, r1, r2, r3, r4, js, 1)
        ev2 = _pc_ev(mc, greens, r1, r2, r3, r4, js, 2)
        ev3 = _pc_ev(mc, greens, r1, r2, r3, r4, js, 3)
        ev4 = _pc_ev(mc, greens, r1, r2, r3, r4, js, 4)

        Pm[y+1,x+1] += ev1 - ev2 - ev3 + ev4
        Pp[y+1,x+1] += ev1 + ev2 + ev3 + ev4
      end
    end
  end

  # r0 mean
  Pm ./= N
  Pp ./= N

  nothing
end






















# -------------------------------------------------------
#         Equal-time Charge Density Correlations
# -------------------------------------------------------
function allocate_etcdc!(mc)
  L = mc.p.L
  meas = mc.s.meas

  meas.etcdc_minus = zeros(Float64,L,L)
  meas.etcdc_plus = zeros(Float64,L,L)

  println("Allocated memory for ETCDC measurements.")
  nothing
end


"""
Calculate equal time charge density susceptibilities

  Cη(y,x) = << Δη^†(r)Δη(0) >> = etcdc_plus/minus(y,x)

where η=± (s and d-wave), r=(y,x), and <<·>> indicates a fermion average for fixed ϕ.

x,y ∈ [0,L-1]

Details:
- we mean over r_0, that is "0".
- we do not mean over τ_0=0 (τ not shown above).
"""
function etcdc!(mc::AbstractDQMC, greens::AbstractMatrix)
  L = mc.p.L
  N = mc.l.sites
  Cm = mc.s.meas.etcdc_minus # d-wave
  Cp = mc.s.meas.etcdc_plus # s-wave

  fill!(Cm, 0.0)
  fill!(Cp, 0.0)

  # xu, yd, xd, yu = 1,2,3,4
  X, Y = 1, 2
  UP, DOWN = 0, 2
  flv = (band, spin) -> band + spin

  sql = reshape(1:N,L,L)

  @inbounds for x0 in 1:L, y0 in 1:L # r0 (origin)
    for s1 in (UP, DOWN), s2 in (UP, DOWN)
      for x in 0:(L-1), y in 0:(L-1) # r (displacement)
        ri = siteidx(mc, sql, x0+x, y0+y)
        r0 = siteidx(mc, sql, x0, y0)

        ev1 = _cdc_ev(mc, greens, greens, greens, greens, ri, ri, r0, r0, flv(X, s1), flv(X, s1), flv(X, s2), flv(X, s2))
        ev2 = _cdc_ev(mc, greens, greens, greens, greens, ri, ri, r0, r0, flv(X, s1), flv(X, s1), flv(Y, s2), flv(Y, s2))
        ev3 = _cdc_ev(mc, greens, greens, greens, greens, ri, ri, r0, r0, flv(Y, s1), flv(Y, s1), flv(X, s2), flv(X, s2))
        ev4 = _cdc_ev(mc, greens, greens, greens, greens, ri, ri, r0, r0, flv(Y, s1), flv(Y, s1), flv(Y, s2), flv(Y, s2))

        Cm[y+1,x+1] += ev1 - ev2 - ev3 + ev4
        Cp[y+1,x+1] += ev1 + ev2 + ev3 + ev4
      end
    end
  end

  # r0 mean
  Cm ./= N
  Cp ./= N

  nothing
end



"""
Charge density correlation expectation value

  etcdc_ev = << c_j1^†(r1) c_j2(r2) c_j3^†(r3) c_j4(r4) >>

(Wick's theorem.)
"""
@inline function _cdc_ev(mc, Gtau, G0, Gt0, G0t, r1, r2, r3, r4, j1, j2, j3, j4)
  N = mc.l.sites
  i1 = greensidx(N, j1, r1)
  i2 = greensidx(N, j2, r2)
  i3 = greensidx(N, j3, r3)
  i4 = greensidx(N, j4, r4)

  uncorr = (kd(i1, i2) - G(mc, i2, i1, Gtau)) * (kd(i3, i4) - G(mc, i4, i3, G0)) # optimize wrt speed?
  corr1 = (kd(i1, i4) - G(mc, i4, i1, G0t))
  corr2 = G(mc, i2, i3, Gt0)

  return uncorr + corr1 * corr2
end





# -------------------------------------------------------
#  Zero-Frequency (static) Charge Density Correlations
# -------------------------------------------------------
function allocate_zfcdc!(mc)
  L = mc.p.L
  meas = mc.s.meas

  meas.zfcdc_minus = zeros(Float64,L,L)
  meas.zfcdc_plus = zeros(Float64,L,L)

  println("Allocated memory for ZFCDC measurements.")
  nothing
end

"""
Calculate zero-frequency charge density susceptibilities

  Cη(y,x) = int_tau << Δη^†(r, tau) Δη(0, 0) >> = zfcdc_plus/minus(y,x)

where η=± (s and d-wave), r=(y,x), and <<·>> indicates a fermion average for fixed ϕ.

x,y ∈ [0,L-1]

The input `greens` can either be a single ETGF or all ETGFs, i.e. G(tau).

Details:
- we mean over r_0, that is "0".
- we do not mean over τ_0=0 (τ not shown above).
"""
function zfcdc!(mc::AbstractDQMC, greens::Union{V, W}, Gt0s::AbstractVector{T}, G0ts::AbstractVector{T}) where {T <: AbstractMatrix, V <: AbstractVector, W <: AbstractMatrix}
  L = mc.p.L
  N = mc.l.sites
  M = mc.p.slices
  Cm = mc.s.meas.zfcdc_minus # d-wave
  Cp = mc.s.meas.zfcdc_plus # s-wave

  fill!(Cm, 0.0)
  fill!(Cp, 0.0)

  # xu, yd, xd, yu = 1,2,3,4
  X, Y = 1, 2
  UP, DOWN = 0, 2
  flv = (band, spin) -> band + spin

  sql = reshape(1:N,L,L)

  if typeof(greens) <: AbstractMatrix
    get_Gtau = (tau) -> greens
    get_G0 = (tau) -> greens
  else
    get_Gtau = (tau) -> greens[tau]
    get_G0 = (tau) -> greens[1]
  end


  @inbounds for tau in 1:M
    Gtau = get_Gtau(tau)
    G0 = get_G0(tau)
    Gt0 = Gt0s[tau]
    G0t = G0ts[tau]

    for x0 in 1:L, y0 in 1:L # r0 (we average over r0)
      for s1 in (UP, DOWN), s2 in (UP, DOWN)
        for x in 0:(L-1), y in 0:(L-1) # r (displacement)
          ri = siteidx(mc, sql, x0+x, y0+y)
          r0 = siteidx(mc, sql, x0, y0)

          ev1 = _cdc_ev(mc, Gtau, G0, Gt0, G0t, ri, ri, r0, r0, flv(X, s1), flv(X, s1), flv(X, s2), flv(X, s2))
          ev2 = _cdc_ev(mc, Gtau, G0, Gt0, G0t, ri, ri, r0, r0, flv(X, s1), flv(X, s1), flv(Y, s2), flv(Y, s2))
          ev3 = _cdc_ev(mc, Gtau, G0, Gt0, G0t, ri, ri, r0, r0, flv(Y, s1), flv(Y, s1), flv(X, s2), flv(X, s2))
          ev4 = _cdc_ev(mc, Gtau, G0, Gt0, G0t, ri, ri, r0, r0, flv(Y, s1), flv(Y, s1), flv(Y, s2), flv(Y, s2))

          Cm[y+1,x+1] += ev1 - ev2 - ev3 + ev4
          Cp[y+1,x+1] += ev1 + ev2 + ev3 + ev4
        end
      end
    end
  end

  # r0 mean
  Cm ./= N
  Cp ./= N

  nothing
end


















# -------------------------------------------------------
#  Zero-Frequency (static) Current-Current Correlations
# -------------------------------------------------------
function allocate_zfccc!(mc)
  L = mc.p.L
  meas = mc.s.meas

  meas.zfccc = zeros(Float64,L,L)

  println("Allocated memory for ZFCCC measurements.")
  nothing
end





"""
Calculate zero-frequency current-current susceptibilities

  Λxx(y,x) = int_tau << j_x(r, tau) j_x(0, 0) >> = zfccc(y,x)

where r=(y,x), and <<·>> indicates a fermion average for fixed ϕ.

x,y ∈ [0,L-1]

The input `greens` can either be a single ETGF or all ETGFs, i.e. G(tau).

Details:
- we mean over r_0, that is "0".
- we do not mean over τ_0=0 (τ not shown above).
"""
function zfccc!(mc::AbstractDQMC, greens::Union{V, W}, Gt0s::AbstractVector{S}, G0ts::AbstractVector{S}) where {S <: AbstractMatrix, V <: AbstractVector, W <: AbstractMatrix}
  L = mc.p.L
  N = mc.l.sites
  M = mc.p.slices
  Lambda = mc.s.meas.zfccc

  fill!(Lambda, 0.0)

  # xu, yd, xd, yu = 1,2,3,4
  X, Y = 1, 2
  UP, DOWN = 0, 2
  flv = (band, spin) -> band + spin

  sql = reshape(1:N,L,L)

  if typeof(greens) <: AbstractMatrix
    get_Gtau = (tau) -> greens
    get_G0 = (tau) -> greens
  else
    get_Gtau = (tau) -> greens[tau]
    get_G0 = (tau) -> greens[1]
  end

  # OPT: Get rid of this "hack" to get the hopping matrix
  T = log(mc.l.hopping_matrix_exp)
  T .*= -2 / mc.p.delta_tau
  # Check: We set tij but in the action/hamiltonian we have -tij.
  # If we really want tij add another * (-1).


  @inbounds for tau in 1:M
    Gtau = get_Gtau(tau)
    G0 = get_G0(tau)
    Gt0 = Gt0s[tau]
    G0t = G0ts[tau]

    # TODO: Maybe we can use symmetries to reduce a1,a2,s1,s2 sum?

    for x0 in 1:L, y0 in 1:L # r0 (we average over r0)
      for a1 in (X, Y), a2 in (X, Y)
        for s1 in (UP, DOWN), s2 in (UP, DOWN)
          for x in 0:(L-1), y in 0:(L-1) # r (displacement)
            ri = siteidx(mc, sql, x0+x, y0+y)
            rj = siteidx(mc, sql, x0+x+1, y0+y) # ri + x̂
            r0 = siteidx(mc, sql, x0, y0)
            r0p = siteidx(mc, sql, x0+1, y0) # r0 + x̂

            ais = greensidx(N, flv(a1, s1), ri)
            ajs = greensidx(N, flv(a1, s1), rj)
            ap0sp = greensidx(N, flv(a2, s2), r0)
            ap0psp = greensidx(N, flv(a2, s2), r0p)

            # hoppings
            tij = T[ais, ajs]
            tji = T[ajs, ais]
            t00p = T[ap0sp, ap0psp]
            t0p0 = T[ap0psp, ap0sp]


            # Uncorrelated part
            # TODO: Check and add real() later
            Lambda[y+1, x+1] += (tij * G(mc, ajs, ais, Gtau) - tji * G(mc, ais, ajs, Gtau)) *
                                (t00p * G(mc, ap0psp, ap0sp, G0) - t0p0 * G(mc, ap0sp, ap0psp, G0))

            # Correlated part
            # TODO: Check and add real() later
            Lambda[y+1, x+1] += - tij * t00p * G(mc, ap0psp, ais, G0t) * G(mc, ajs, ap0sp, Gt0) +
                                + tij * t0p0 * G(mc, ap0sp, ais, G0t) * G(mc, ajs, ap0psp, Gt0) +
                                + tji * t00p * G(mc, ap0psp, ajs, G0t) * G(mc, ais, ap0sp, Gt0) +
                                - tji * t0p0 * G(mc, ap0sp, ajs, G0t) * G(mc, ais, ap0psp, Gt0)
          end
        end
      end
    end
  end

  # r0 mean + overall minus sign due to i^2
  Lambda ./= -N

  nothing
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
  _calc_greens_helper(mc, slice, safe_mult)

  rmul!(s.U, Diagonal(s.d))
  return s.U * s.T
end
function calc_greens_and_logdet(mc::AbstractDQMC, slice::Int=mc.s.current_slice, safe_mult::Int=mc.p.safe_mult)
  s = mc.s
  _calc_greens_helper(mc, slice, safe_mult)

  ldet = real(log(complex(det(s.U))) + sum(log.(s.d)) + log(complex(det(s.T))))

  rmul!(s.U, Diagonal(s.d))
    return s.U * s.T, ldet
end


# Result in s.U, s.d, and s.T
function _calc_greens_helper(mc::AbstractDQMC, slice::Int, safe_mult::Int)
  flv = mc.p.flv
  N = mc.l.sites
  s = mc.s
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

  inv_one_plus_two_udts!(mc, s.U,s.d,s.T, Ul, Dl, Tl, Ur,Dr,Tr)

  nothing
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
@enum Direction begin
  LEFT
  RIGHT
end

function allocate_tdgfs!(mc)
  @stackshortcuts
  M = mc.p.slices
  Nflv = N*flv
  meas = mc.s.meas

  nranges = length(mc.s.ranges)

  meas.Gt0 = Matrix{G}[zeros(G, Nflv, Nflv) for _ in 1:M]
  meas.G0t = Matrix{G}[zeros(G, Nflv, Nflv) for _ in 1:M]

  mc.s.meas.BT0Inv_u_stack = Matrix{G}[zeros(G, flv*N, flv*N) for _ in 1:nranges]
  mc.s.meas.BT0Inv_d_stack = Vector{Float64}[zeros(Float64, flv*N) for _ in 1:nranges]
  mc.s.meas.BT0Inv_t_stack = Matrix{G}[zeros(G, flv*N, flv*N) for _ in 1:nranges]
  mc.s.meas.BBetaT_u_stack = Matrix{G}[zeros(G, flv*N, flv*N) for _ in 1:nranges]
  mc.s.meas.BBetaT_d_stack = Vector{Float64}[zeros(Float64, flv*N) for _ in 1:nranges]
  mc.s.meas.BBetaT_t_stack = Matrix{G}[zeros(G, flv*N, flv*N) for _ in 1:nranges]
  mc.s.meas.BT0_u_stack = Matrix{G}[zeros(G, flv*N, flv*N) for _ in 1:nranges]
  mc.s.meas.BT0_d_stack = Vector{Float64}[zeros(Float64, flv*N) for _ in 1:nranges]
  mc.s.meas.BT0_t_stack = Matrix{G}[zeros(G, flv*N, flv*N) for _ in 1:nranges]
  mc.s.meas.BBetaTInv_u_stack = Matrix{G}[zeros(G, flv*N, flv*N) for _ in 1:nranges]
  mc.s.meas.BBetaTInv_d_stack = Vector{Float64}[zeros(Float64, flv*N) for _ in 1:nranges]
  mc.s.meas.BBetaTInv_t_stack = Matrix{G}[zeros(G, flv*N, flv*N) for _ in 1:nranges]

  println("Allocated memory for TDGF measurement.")
  nothing
end

function deallocate_tdgfs_stacks!(mc)

  mc.s.meas.BT0Inv_u_stack = Matrix{G}[]
  mc.s.meas.BT0Inv_d_stack = Vector{Float64}[]
  mc.s.meas.BT0Inv_t_stack = Matrix{G}[]
  mc.s.meas.BBetaT_u_stack = Matrix{G}[]
  mc.s.meas.BBetaT_d_stack = Vector{Float64}[]
  mc.s.meas.BBetaT_t_stack = Matrix{G}[]
  mc.s.meas.BT0_u_stack = Matrix{G}[]
  mc.s.meas.BT0_d_stack = Vector{Float64}[]
  mc.s.meas.BT0_t_stack = Matrix{G}[]
  mc.s.meas.BBetaTInv_u_stack = Matrix{G}[]
  mc.s.meas.BBetaTInv_d_stack = Vector{Float64}[]
  mc.s.meas.BBetaTInv_t_stack = Matrix{G}[]

  println("Deallocated UDT stacks memory of TDGF measurement.")
  nothing
end


# TODO: Comment!
function calc_tdgfs!(mc)
  G = geltype(mc)
  M = mc.p.slices
  N = mc.l.sites
  flv = mc.p.flv
  Nflv = N * flv
  safe_mult = mc.p.safe_mult
  eye_full = mc.s.eye_full
  ones_vec = mc.s.ones_vec
  nranges = length(mc.s.ranges)

  Gt0 = mc.s.meas.Gt0
  G0t = mc.s.meas.G0t

  BT0Inv_u_stack = mc.s.meas.BT0Inv_u_stack
  BT0Inv_d_stack = mc.s.meas.BT0Inv_d_stack
  BT0Inv_t_stack = mc.s.meas.BT0Inv_t_stack
  BBetaT_u_stack = mc.s.meas.BBetaT_u_stack
  BBetaT_d_stack = mc.s.meas.BBetaT_d_stack
  BBetaT_t_stack = mc.s.meas.BBetaT_t_stack
  BT0_u_stack = mc.s.meas.BT0_u_stack
  BT0_d_stack = mc.s.meas.BT0_d_stack
  BT0_t_stack = mc.s.meas.BT0_t_stack
  BBetaTInv_u_stack = mc.s.meas.BBetaTInv_u_stack
  BBetaTInv_d_stack = mc.s.meas.BBetaTInv_d_stack
  BBetaTInv_t_stack = mc.s.meas.BBetaTInv_t_stack
  

  # ---- first, calculate Gt0 and G0t only at safe_mult slices 
  # right mult (Gt0)
  calc_Bchain_udts!(mc, BT0Inv_u_stack, BT0Inv_d_stack, BT0Inv_t_stack, invert=true, dir=LEFT);
  calc_Bchain_udts!(mc, BBetaT_u_stack, BBetaT_d_stack, BBetaT_t_stack, invert=false, dir=RIGHT);
  
  # left mult (G0t)
  calc_Bchain_udts!(mc, BT0_u_stack, BT0_d_stack, BT0_t_stack, invert=false, dir=LEFT);
  calc_Bchain_udts!(mc, BBetaTInv_u_stack, BBetaTInv_d_stack, BBetaTInv_t_stack, invert=true, dir=RIGHT);


  safe_mult_taus = 1:safe_mult:mc.p.slices
  @inbounds for i in 1:length(safe_mult_taus) # i = ith safe mult time slice
    tau = safe_mult_taus[i] # tau = tauth (overall) time slice
    if i != 1
      # Gt0
      inv_sum_udts_scalettar!(mc, Gt0[tau], BT0Inv_u_stack[i-1], BT0Inv_d_stack[i-1], BT0Inv_t_stack[i-1],
                   BBetaT_u_stack[i], BBetaT_d_stack[i], BBetaT_t_stack[i]) # G(i,0) = G(mc.s.ranges[i][1], 0), i.e. G(21, 1) for i = 3
      effective_greens2greens!(mc, Gt0[tau])

      # G0t
      inv_sum_udts_scalettar!(mc, G0t[tau], BT0_u_stack[i-1], BT0_d_stack[i-1], BT0_t_stack[i-1],
                   BBetaTInv_u_stack[i], BBetaTInv_d_stack[i], BBetaTInv_t_stack[i]) # G(i,0) = G(mc.s.ranges[i][1], 0), i.e. G(21, 1) for i = 3
      effective_greens2greens!(mc, G0t[tau])
    else
      # Gt0
      inv_one_plus_udt_scalettar!(mc, Gt0[tau], BBetaT_u_stack[1], BBetaT_d_stack[1], BBetaT_t_stack[1])
      effective_greens2greens!(mc, Gt0[tau])

      # G0t
      inv_one_plus_udt_scalettar!(mc, G0t[tau], BBetaTInv_u_stack[1], BBetaTInv_d_stack[1], BBetaTInv_t_stack[1])
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


"""
Calculate UDTs at safe_mult time slices of
dir = LEFT: 
inv=false:  B(tau, 1) = B(tau) * B(tau-1) * ... * B(1)                    # mult left, 1:tau
inv=true:   [B(tau, 1)]^-1 = B(1)^-1 * B(2)^-1 * ... B(tau)^-1            # mult inv right, 1:tau

udv[i] = from 1 to mc.s.ranges[i][end]


dir = RIGHT:
inv=false:  B(beta, tau) = B(beta) * B(beta-1) * ... * B(tau)             # mult right, beta:tau
inv=true:   [B(beta, tau)]^-1 = B(tau)^-1 * B(tau+1)^-1 * ... B(beta)^-1  # mult inv left, beta:tau

udv[i] = from mc.s.ranges[i][1] to mc.p.slices (beta)
"""
function calc_Bchain_udts!(mc::AbstractDQMC, u_stack, d_stack, t_stack; invert::Bool=false, dir::Direction=LEFT)
  G = geltype(mc)
  flv = mc.p.flv
  N = mc.l.sites
  nranges= length(mc.s.ranges)
  curr_U_or_T = mc.s.curr_U
  eye_full = mc.s.eye_full
  ones_vec = mc.s.ones_vec
  ranges = mc.s.ranges

  rightmult = false
  ((dir == RIGHT && !invert) || (dir == LEFT && invert)) && (rightmult = true)

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
      if invert == false
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

  if dir == RIGHT
    reverse!(u_stack), reverse!(d_stack), reverse!(t_stack)
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










# Calculate "G(tau, 0)", i.e. G(slice,1) as G(slice,1) = [B(slice, 1)^-1 + B(beta, slice)]^-1 which is equal to B(slice,1)G(1)
function calc_tdgf_direct(mc::AbstractDQMC, slice::Int, safe_mult::Int=mc.p.safe_mult; scalettar=true)
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
  if scalettar
    U, D, T = inv_sum_udts(Ul, Dl, Tl, Ur, Dr, Tr)
  else
    U, D, T = inv_sum_udts_scalettar(Ul, Dl, Tl, Ur, Dr, Tr)
  end
  effective_greens2greens!(mc, U, T)

  rmul!(U, Diagonal(D))
  return U*T
end


function test_Gt0(mc::AbstractDQMC)
  Nflv = mc.l.sites * mc.p.flv
  G = geltype(mc)
  s = mc.s

  eye_full = mc.s.eye_full
  ones_vec = mc.s.ones_vec

  Gt0 = fill(zero(G), Nflv, Nflv)

  # i = 3
  # i = 11 # == 101, almost beta half = 100
  i = 1

  if i != 1
    U,D,T = inv_sum_udts(s.BT0Inv_u_stack[i-1], s.BT0Inv_d_stack[i-1], s.BT0Inv_t_stack[i-1],
                 s.BBetaT_u_stack[i], s.BBetaT_d_stack[i], s.BBetaT_t_stack[i])
    UDT_to_mat!(Gt0, U, D, T) # G(i,0) = G(mc.s.ranges[i][1], 0), i.e. G(21, 1) for i = 3
    effective_greens2greens!(mc, Gt0)
  else
    U,D,T = inv_sum_udts(eye_full, ones_vec, eye_full,
                 s.BBetaT_u_stack[i], s.BBetaT_d_stack[i], s.BBetaT_t_stack[i])
    UDT_to_mat!(Gt0, U, D, T) # G(i,0) = G(mc.s.ranges[i][1], 0), i.e. G(21, 1) for i = 3
    effective_greens2greens!(mc, Gt0)
  end

  tdgf = calc_tdgf_direct(mc, mc.s.ranges[i][1]);
  compare(tdgf, Gt0)

  # compare G(0,0) with G(0) for i=1
  g = calc_greens(mc, 1);
  compare(Gt0, g) # 1e-16 for i=1


  # test all Gt0 at safe mult slices
  safe_mult_taus = 1:safe_mult:mc.p.slices
  for tau in safe_mult_taus
    i = ceil(Int, tau/safe_mult)
    tdgf = calc_tdgf_direct(mc, mc.s.ranges[i][1]);
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










































# SVD STUFF ----------------------------------------------------
"""
SVD DECOMPOSITION: Calculate effective(!) Green's function (direct, i.e. without stack)
"""
function calc_Bchain_udv(mc::AbstractDQMC, start::Int, stop::Int, safe_mult::Int=mc.p.safe_mult)
  # Returns: tuple of results (U, D, and V) and log singular values of the intermediate products
  flv = mc.p.flv
  N = mc.l.sites
  G = geltype(mc)

  @assert 0 < start <= mc.p.slices
  @assert 0 < stop <= mc.p.slices
  @assert start <= stop

  U = Matrix{G}(I, flv*N, flv*N)
  D = ones(Float64, flv*N)
  Vt = Matrix{G}(I, flv*N, flv*N)

  svs = zeros(flv*N,length(start:stop))
  svc = 1
  for k in start:stop
    if mod(k,safe_mult) == 0 || k == stop # always decompose in the end
      multiply_B_left!(mc,k,U)
      rmul!(U, Diagonal(D))
      U, D, Vtnew = decompose_udv(U)
      mul!(mc.s.tmp, Vtnew, Vt)
      Vt .=  mc.s.tmp
      svs[:,svc] = log.(D)
      svc += 1
    else
      multiply_B_left!(mc,k,U)
    end
  end
  return (U,D,Vt,svs)
end


function calc_Bchain_inv_udv(mc::AbstractDQMC, start::Int, stop::Int, safe_mult::Int=mc.p.safe_mult)
  # Calculate Ul, Dl, Vtl = [B(stop) ... B(start)]^(-1) = B(start)^(-1) ... B(stop)^(-1)
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
  for k in reverse(start:stop)
    if mod(k,safe_mult) == 0
      multiply_B_inv_left!(mc,k,U)
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



"""
Calculate ETGF (slice=1) as inv_one_plus_udv(calc_Bchain_udv)
"""
function calc_greens_udv_simple(mc::AbstractDQMC)
  Budv = calc_Bchain_udv(mc, 1, mc.p.slices);
  gudv = inv_one_plus_udv_scalettar(Budv[1], Budv[2], Budv[3])
end