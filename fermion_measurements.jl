if !isdefined(:GreensType)
  global const GreensType = Complex128; # assume O(2) or O(3)
  warn("GreensType wasn't set on loading fermion_measurements.jl")
  println("GreensType = ", GreensType)
end

# -------------------------------------------------------
#         Measurements
# -------------------------------------------------------
function measure_greens_and_logdet(mc::AbstractDQMC, safe_mult::Int=mc.p.safe_mult)
  greens, greens_logdet = calculate_greens_and_logdet(mc, 1, safe_mult)
  return effective_greens2greens!(mc, greens), greens_logdet
end

function measure_greens(mc::AbstractDQMC, safe_mult::Int=mc.p.safe_mult)
  return measure_greens_and_logdet(mc, safe_mult)[1]
end

# -------------------------------------------------------
#         Postprocessing/Analysis
# -------------------------------------------------------
# Go from xup, ydown, xdown, yup -> xup, yup, xdown, ydown
function permute_greens(greens::AbstractArray{GreensType, 2})
  const perm = [1,4,3,2] # flv*spin: xup, ydown, xdown, yup -> xup, yup, xdown, ydown
  const N = Int(sqrt(length(greens))/4)
  return reshape(reshape(greens, (N,4,N,4))[:,perm,:,perm], (4*N,4*N)); # rfs1, rfs2
end

# Assuming translational invariance go to momentum space Greens function (k, fs1, fs2)
using PyCall
@pyimport numpy as np
function fft_greens(greens::AbstractArray{GreensType})
  const L = Int(sqrt(sqrt(length(greens))/4))
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
SVD DECOMPOSITION: Calculate effective(!) Green's function (direct, i.e. without stack)
"""
# Calculate B(stop) ... B(start) safely (with stabilization at every safe_mult step, default ALWAYS)
# Returns: tuple of results (U, D, and V) and log singular values of the intermediate products
function calculate_slice_matrix_chain_udv(mc::AbstractDQMC, start::Int, stop::Int, safe_mult::Int=mc.p.safe_mult)
  const flv = mc.p.flv
  const slices = mc.p.slices
  const N = mc.l.sites

  @assert 0 < start <= slices
  @assert 0 < stop <= slices
  @assert start <= stop

  U = eye(GreensType, flv*N, flv*N)
  D = ones(Float64, flv*N)
  Vt = eye(GreensType, flv*N, flv*N)
  Vtnew = eye(GreensType, flv*N, flv*N)

  svs = zeros(flv*N,length(start:stop))
  svc = 1
  for k in start:stop
    if mod(k,safe_mult) == 0
      multiply_slice_matrix_left!(mc,k,U)
      U *= spdiagm(D)
      U, D, Vtnew = decompose_udv(U)
      Vt =  Vtnew * Vt
      svs[:,svc] = log.(D)
      svc += 1
    else
      multiply_slice_matrix_left!(mc,k,U)
    end
  end
  return (U,D,Vt,svs)
end

# Calculate G(slice) = [1+B(slice-1)...B(1)B(M) ... B(slice)]^(-1) and its logdet in a stable manner
function calculate_greens_and_logdet_udv(mc::AbstractDQMC, slice::Int, safe_mult::Int=mc.p.safe_mult)
  const flv = mc.p.flv
  const N = mc.l.sites

  # Calculate Ur,Dr,Vtr=B(M) ... B(slice)
  Ur, Dr, Vtr = calculate_slice_matrix_chain_udv(mc,slice,mc.p.slices,safe_mult)

  # Calculate Ul,Dl,Vtl=B(slice-1) ... B(1)
  if slice-1 >= 1
    Ul, Dl, Vtl = calculate_slice_matrix_chain_udv(mc,1,slice-1,safe_mult)
  else
    Ul = eye(GreensType, flv*N)
    Dl = ones(Float64, flv*N)
    Vtl = eye(GreensType, flv*N)
  end

  # Calculate Greens function
  tmp = Vtl * Ur
  inner = ctranspose(Vtr * Ul) + spdiagm(Dl) * tmp * spdiagm(Dr)
  I = decompose_udv!(inner)
  U = ctranspose(I[3] * Vtr)
  D = spdiagm(1./I[2])
  Vt = ctranspose(Ul * I[1])
  return U*D*Vt, sum(log.(diag(D)))
end


"""
QR DECOMPOSITION: Calculate effective(!) Green's function (direct, i.e. without stack)
"""
# Calculate Ul, Dl, Tl =B(stop) ... B(start)
function calculate_slice_matrix_chain_qr(mc::AbstractDQMC, start::Int, stop::Int, safe_mult::Int=mc.p.safe_mult)
  const flv = mc.p.flv
  const N = mc.l.sites

  @assert 0 < start <= mc.p.slices
  @assert 0 < stop <= mc.p.slices
  @assert start <= stop

  U = eye(GreensType, flv*N, flv*N)
  D = ones(Float64, flv*N)
  T = eye(GreensType, flv*N, flv*N)
  Tnew = eye(GreensType, flv*N, flv*N)

  svs = zeros(flv*N,length(start:stop))
  svc = 1
  for k in start:stop
    if mod(k,safe_mult) == 0
      multiply_slice_matrix_left!(mc,k,U)
      U *= spdiagm(D)
      U, D, Tnew = decompose_udt(U)
      T =  Tnew * T
      svs[:,svc] = log.(D)
      svc += 1
    else
      multiply_slice_matrix_left!(mc,k,U)
    end
  end
  return (U,D,T,svs)
end

# Calculate (Ur, Dr, Tr)' = B(stop) ... B(start)  => Ur,Dr, Tr = B(start)' ... B(stop)'
function calculate_slice_matrix_chain_qr_dagger(mc::AbstractDQMC, start::Int, stop::Int, safe_mult::Int=mc.p.safe_mult)
  const flv = mc.p.flv
  const N = mc.l.sites

  @assert 0 < start <= mc.p.slices
  @assert 0 < stop <= mc.p.slices
  @assert start <= stop

  U = eye(GreensType, flv*N, flv*N)
  D = ones(Float64, flv*N)
  T = eye(GreensType, flv*N, flv*N)
  Tnew = eye(GreensType, flv*N, flv*N)

  svs = zeros(flv*N,length(start:stop))
  svc = 1
  for k in reverse(start:stop)
    if mod(k,safe_mult) == 0
      multiply_daggered_slice_matrix_left!(mc,k,U)
      U *= spdiagm(D)
      U, D, Tnew = decompose_udt(U)
      T =  Tnew * T
      svs[:,svc] = log.(D)
      svc += 1
    else
      multiply_daggered_slice_matrix_left!(mc,k,U)
    end
  end
  return (U,D,T,svs)
end

# Calculate G(slice) = [1+B(slice-1)...B(1)B(M) ... B(slice)]^(-1) and its singular values in a stable manner
function calculate_greens_and_logdet(mc::AbstractDQMC, slice::Int, safe_mult::Int=mc.p.safe_mult)
  const flv = mc.p.flv
  const N = mc.l.sites

  # Calculate Ur,Dr,Tr=B(slice)' ... B(M)'
  Ur, Dr, Tr = calculate_slice_matrix_chain_qr_dagger(mc,slice,mc.p.slices, safe_mult)

  # Calculate Ul,Dl,Tl=B(slice-1) ... B(1)
  if slice-1 >= 1
    Ul, Dl, Tl = calculate_slice_matrix_chain_qr(mc,1,slice-1, safe_mult)
  else
    Ul = eye(GreensType, flv*N)
    Dl = ones(Float64, flv*N)
    Tl = eye(GreensType, flv*N)
  end

  tmp = Tl * ctranspose(Tr)
  U, D, T = decompose_udt(spdiagm(Dl) * tmp * spdiagm(Dr))
  U = Ul * U
  T *= ctranspose(Ur)

  u, d, t = decompose_udt(ctranspose(U) * inv(T) + spdiagm(D))

  T = inv(t * T)
  U *= u
  U = ctranspose(U)
  d = 1./d

  ldet = real(log(complex(det(U))) + sum(log.(d)) + log(complex(det(T))))

  return T * spdiagm(d) * U, ldet
end


# -------------------------------------------------------
#    Effective Green's function -> Green's function
# -------------------------------------------------------
function effective_greens2greens!(mc::DQMC_CBTrue, greens::AbstractArray{GreensType, 2})
  const chkr_hop_half_minus = mc.l.chkr_hop_half
  const chkr_hop_half_plus = mc.l.chkr_hop_half_inv
  const n_groups = mc.l.n_groups

  @inbounds @views begin
      for i in reverse(1:n_groups)
        greens .= greens * chkr_hop_half_minus[i]
      end
      for i in reverse(1:n_groups)
        greens .= chkr_hop_half_plus[i] * greens
      end
  end
  nothing
end
function greens2effective_greens!(mc::DQMC_CBTrue, greens::AbstractArray{GreensType, 2})
  const chkr_hop_half_minus = mc.l.chkr_hop_half
  const chkr_hop_half_plus = mc.l.chkr_hop_half_inv
  const n_groups = mc.l.n_groups

  @inbounds @views begin
      for i in 1:n_groups
        greens .= greens * chkr_hop_half_plus[i]
      end
      for i in 1:n_groups
        greens .= chkr_hop_half_minus[i] * greens
      end
  end
  nothing
end
function effective_greens2greens!(mc::DQMC_CBFalse, greens::AbstractArray{GreensType, 2})
  const eTminus = mc.l.hopping_matrix_exp
  const eTplus = mc.l.hopping_matrix_exp_inv

  greens .= greens * eTminus
  greens .= eTplus * greens
  nothing
end
function effective_greens2greens!(mc::DQMC_CBFalse, U::AbstractArray{GreensType, 2}, T::AbstractArray{GreensType, 2})
  const eTminus = mc.l.hopping_matrix_exp
  const eTplus = mc.l.hopping_matrix_exp_inv

  T .= T * eTminus
  U .= eTplus * U
  nothing
end
function greens2effective_greens!(mc::DQMC_CBFalse, greens::AbstractArray{GreensType, 2})
  const eTminus = mc.l.hopping_matrix_exp
  const eTplus = mc.l.hopping_matrix_exp_inv

  greens .= greens * eTplus
  greens .= eTminus * greens
  nothing
end
function effective_greens2greens(mc::AbstractDQMC, greens::AbstractArray{GreensType, 2})
  g = copy(greens)
  effective_greens2greens!(mc, g)
  return g
end



# -------------------------------------------------------
#         Time-displaced Green's function
# -------------------------------------------------------
# Calculate "G(tau, 0)", i.e. G(slice,1) naively
function calculate_tdgf_naive(mc::AbstractDQMC, slice::Int)
  G, = calculate_greens_and_logdet(mc,1,2*mc.p.slices)
  Ul, Dl, Tl = calculate_slice_matrix_chain_qr(mc,1,slice,2*mc.p.slices)
  B12 = Ul * spdiagm(Dl) * Tl
  return B12 * G
end

# Calculate "G(tau, 0)", i.e. G(slice,1)
function calculate_tdgf(mc::AbstractDQMC, slice::Int, safe_mult::Int=mc.p.safe_mult)
  const flv = mc.p.flv
  const N = mc.l.sites

  # Calculate Ur,Dr,Tr=B(slice)' ... B(M)'
  Ur, Dr, Tr = calculate_slice_matrix_chain_qr_dagger(s,p,l,slice,mc.p.slices, safe_mult)

  # Calculate Ul,Dl,Tl=B(slice-1) ... B(1)
  if slice-1 >= 1
    Ul, Dl, Tl = calculate_slice_matrix_chain_qr(s,p,l,1,slice-1, safe_mult)
  else
    Ul = eye(GreensType, flv*N)
    Dl = ones(Float64, flv*N)
    Tl = eye(GreensType, flv*N)
  end

  tmp = Tl * ctranspose(Tr)
  U, D, T = decompose_udt(spdiagm(Dl) * tmp * spdiagm(Dr))
  U = Ul * U
  T *= ctranspose(Ur)

  u, d, t = decompose_udt(ctranspose(U) * inv(T) + spdiagm(D))

  T = inv(t * T)
  U *= u
  U = ctranspose(U)
  d = 1./d

  # time displace
  Ul, Dl, Tl = calculate_slice_matrix_chain_qr(mc,1,slice, safe_mult)
  # effective -> actual
  # doesn't seem to change result at all
  effective_greens2greens!(mc,Ul,Tl)

  # effective -> actual (ultra safe)
  # doesn't seem to change result at all (1e-13)
  # UT, DT, TT = decompose_udt(l.hopping_matrix_exp)
  # UTi, DTi, TTi = decompose_udt(l.hopping_matrix_exp_inv)
  # Ul, Dl, Tl = multiply_safely(Ul, Dl, Tl, UT, DT, TT)
  # Ul, Dl, Tl = multiply_safely(UTi, DTi, TTi, Ul, Dl, Tl)

  U, D, T = multiply_safely(Ul, Dl, Tl, T, d, U)

  return U*spdiagm(D)*T
end