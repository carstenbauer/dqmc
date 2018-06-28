# -------------------------------------------------------
#         Measurements
# -------------------------------------------------------
function measure_greens_and_logdet(mc::AbstractDQMC, safe_mult::Int=mc.p.safe_mult)
  greens, greens_logdet = calc_greens_and_logdet(mc, 1, safe_mult)
  effective_greens2greens!(mc, greens)
  return greens, greens_logdet
end

function measure_greens(mc::AbstractDQMC, safe_mult::Int=mc.p.safe_mult)
  return measure_greens_and_logdet(mc, safe_mult)[1]
end

"""
Average occupation per site and fermion flavor.

Example: half-filling of spin 1/2 fermions on a lattice corresponds to `n=0.5`.
"""
function occupation(mc::AbstractDQMC, greens::AbstractMatrix=mc.s.greens)
  n = mean(1 .- diag(greens))
  # only per site:
  # const N = mc.l.sites
  # const M = mc.p.slices
  # n = 1/(N*M) * sum(1 .- diag(greens))
  real(n)
end

# -------------------------------------------------------
#         Postprocessing/Analysis
# -------------------------------------------------------
# Go from xup, ydown, xdown, yup -> xup, yup, xdown, ydown
function permute_greens(greens::AbstractMatrix)
  const perm = [1,4,3,2] # flv*spin: xup, ydown, xdown, yup -> xup, yup, xdown, ydown
  const N = Int(sqrt(length(greens))/4)
  return reshape(reshape(greens, (N,4,N,4))[:,perm,:,perm], (4*N,4*N)); # rfs1, rfs2
end

# Assuming translational invariance go to momentum space Greens function (k, fs1, fs2)
try
  eval(Expr(:toplevel, Expr(:using, Symbol("PyCall"))))
  eval(Expr(:toplevel, parse("@pyimport numpy as np")))
end
function fft_greens(greens::AbstractMatrix)
  if !isdefined(:PyCall)
    eval(Expr(:toplevel, Expr(:using, Symbol("PyCall"))))
    eval(Expr(:toplevel, parse("@pyimport numpy as np")))
    println("Loaded PyCall. Please execute again.")
    return
  end

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
QR DECOMPOSITION: Calculate effective(!) Green's function (direct, i.e. without stack)
"""
# Calculate Ul, Dl, Tl =B(stop) ... B(start)
function calc_Bchain(mc::AbstractDQMC, start::Int, stop::Int, safe_mult::Int=mc.p.safe_mult)
  const flv = mc.p.flv
  const N = mc.l.sites
  const G = geltype(mc)

  @assert 0 < start <= mc.p.slices
  @assert 0 < stop <= mc.p.slices
  @assert start <= stop

  U = eye(G, flv*N, flv*N)
  D = ones(Float64, flv*N)
  T = eye(G, flv*N, flv*N)

  svs = zeros(flv*N,length(start:stop))
  svc = 1
  for k in start:stop
    if mod(k,safe_mult) == 0 || k == stop # always decompose in the end
      multiply_B_left!(mc,k,U)
      scale!(U, D)
      U, Tnew = decompose_udt!(U, D)
      A_mul_B!(mc.s.tmp, Tnew, T)
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
  const flv = mc.p.flv
  const N = mc.l.sites
  const G = geltype(mc)

  @assert 0 < start <= mc.p.slices
  @assert 0 < stop <= mc.p.slices
  @assert start <= stop

  U = eye(G, flv*N, flv*N)
  D = ones(Float64, flv*N)
  T = eye(G, flv*N, flv*N)

  svs = zeros(flv*N,length(start:stop))
  svc = 1
  for k in reverse(start:stop)
    if mod(k,safe_mult) == 0 || k == start # always decompose in the end
      multiply_B_inv_left!(mc,k,U)
      scale!(U, D)
      U, Tnew = decompose_udt!(U, D)
      A_mul_B!(mc.s.tmp, Tnew, T)
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
  const flv = mc.p.flv
  const N = mc.l.sites
  const G = geltype(mc)

  @assert 0 < start <= mc.p.slices
  @assert 0 < stop <= mc.p.slices
  @assert start <= stop

  U = eye(G, flv*N, flv*N)
  D = ones(Float64, flv*N)
  T = eye(G, flv*N, flv*N)

  svs = zeros(flv*N,length(start:stop))
  svc = 1
  for k in reverse(start:stop)
    if mod(k,safe_mult) == 0 || k == start # always decompose in the end
      multiply_daggered_B_left!(mc,k,U)
      scale!(U, D)
      U, Tnew = decompose_udt!(U, D)
      A_mul_B!(mc.s.tmp, Tnew, T)
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
function calc_greens(mc::AbstractDQMC, slice::Int, safe_mult::Int=mc.p.safe_mult)
  const s = mc.s
  calc_greens_helper(mc, slice, safe_mult)

  scale!(s.d, s.U)
  return s.T * s.U
end
function calc_greens_and_logdet(mc::AbstractDQMC, slice::Int, safe_mult::Int=mc.p.safe_mult)
  const s = mc.s
  calc_greens_helper(mc, slice, safe_mult)

  ldet = real(log(complex(det(s.U))) + sum(log.(s.d)) + log(complex(det(s.T))))

  scale!(s.d, s.U)
  return s.T * s.U, ldet
end
function calc_greens_udt(mc::AbstractDQMC, slice::Int, safe_mult::Int=mc.p.safe_mult)
  const s = mc.s
  calc_greens_helper(mc, slice, safe_mult)

  # greens = s.T * spdiagm(s.d) * s.U
  scale!(s.d, s.U)
  U, T = decompose_udt!(s.U, s.d)
  return s.T*U, copy(s.d), T
end

# result in s.T, s.U and s.d
function calc_greens_helper(mc::AbstractDQMC, slice::Int, safe_mult::Int)
  const flv = mc.p.flv
  const N = mc.l.sites
  const G = geltype(mc)

  # Calculate Ur,Dr,Tr=B(slice)' ... B(M)'
  Ur, Dr, Tr = calc_Bchain_dagger(mc,slice,mc.p.slices, safe_mult)

  # Calculate Ul,Dl,Tl=B(slice-1) ... B(1)
  if slice-1 >= 1
    Ul, Dl, Tl = calc_Bchain(mc,1,slice-1, safe_mult)
  else
    Ul = eye(G, flv*N)
    Dl = ones(Float64, flv*N)
    Tl = eye(G, flv*N)
  end

  # calculate greens
  const s = mc.s
  const tmp = mc.s.tmp
  const tmp2 = mc.s.tmp2

  A_mul_Bc!(tmp, Tl, Tr)
  scale!(tmp, Dr)
  scale!(Dl, tmp)
  s.U, s.T = decompose_udt!(tmp, s.D)

  A_mul_B!(tmp, Ul, s.U)
  s.U .= tmp
  A_mul_Bc!(tmp2, s.T, Ur)
  s.T .= tmp2
  Ac_mul_B!(tmp, s.U, inv(s.T))
  tmp[diagind(tmp)] .+= s.D
  u, t = decompose_udt!(tmp, s.d)

  A_mul_B!(tmp, t, s.T)
  s.T = inv(tmp)
  A_mul_B!(tmp, s.U, u)
  s.U = ctranspose(tmp)
  s.d .= 1./s.d

  nothing
end



"""
SVD DECOMPOSITION: Calculate effective(!) Green's function (direct, i.e. without stack)
"""
# Calculate B(stop) ... B(start) safely (with stabilization at every safe_mult step, default ALWAYS)
# Returns: tuple of results (U, D, and V) and log singular values of the intermediate products
function calc_Bchain_udv(mc::AbstractDQMC, start::Int, stop::Int, safe_mult::Int=mc.p.safe_mult)
  const flv = mc.p.flv
  const slices = mc.p.slices
  const N = mc.l.sites
  const G = geltype(mc)

  @assert 0 < start <= slices
  @assert 0 < stop <= slices
  @assert start <= stop

  U = eye(G, flv*N, flv*N)
  D = ones(Float64, flv*N)
  Vt = eye(G, flv*N, flv*N)
  Vtnew = eye(G, flv*N, flv*N)

  svs = zeros(flv*N,length(start:stop))
  svc = 1
  for k in start:stop
    if mod(k,safe_mult) == 0
      multiply_B_left!(mc,k,U)
      U *= spdiagm(D)
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
  const flv = mc.p.flv
  const N = mc.l.sites
  const G = geltype(mc)

  # Calculate Ur,Dr,Vtr=B(M) ... B(slice)
  Ur, Dr, Vtr = calc_Bchain_udv(mc,slice,mc.p.slices,safe_mult)

  # Calculate Ul,Dl,Vtl=B(slice-1) ... B(1)
  if slice-1 >= 1
    Ul, Dl, Vtl = calc_Bchain_udv(mc,1,slice-1,safe_mult)
  else
    Ul = eye(G, flv*N)
    Dl = ones(Float64, flv*N)
    Vtl = eye(G, flv*N)
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

# -------------------------------------------------------
#    Effective Green's function -> Green's function
# -------------------------------------------------------
function effective_greens2greens!(mc::DQMC_CBTrue, greens::AbstractMatrix)
  const chkr_hop_half_minus = mc.l.chkr_hop_half
  const chkr_hop_half_plus = mc.l.chkr_hop_half_inv
  const n_groups = mc.l.n_groups
  const tmp = mc.s.tmp

  @inbounds @views begin
      for i in reverse(1:n_groups)
        A_mul_B!(tmp, greens, chkr_hop_half_minus[i])
        greens .= tmp
      end
      for i in reverse(1:n_groups)
        A_mul_B!(tmp, chkr_hop_half_plus[i], greens)
        greens .= tmp
      end
  end
  nothing
end
function effective_greens2greens!(mc::DQMC_CBTrue, U::AbstractMatrix, T::AbstractMatrix)
  const chkr_hop_half_minus = mc.l.chkr_hop_half
  const chkr_hop_half_plus = mc.l.chkr_hop_half_inv
  const n_groups = mc.l.n_groups
  const tmp = mc.s.tmp

  @inbounds @views begin
      for i in reverse(1:n_groups)
        A_mul_B!(tmp, T, chkr_hop_half_minus[i])
        T .= tmp
      end
      for i in reverse(1:n_groups)
        A_mul_B!(tmp, chkr_hop_half_plus[i], U)
        U .= tmp
      end
  end
  nothing
end
function greens2effective_greens!(mc::DQMC_CBTrue, greens::AbstractMatrix)
  const chkr_hop_half_minus = mc.l.chkr_hop_half
  const chkr_hop_half_plus = mc.l.chkr_hop_half_inv
  const n_groups = mc.l.n_groups

  @inbounds @views begin
      for i in 1:n_groups
        A_mul_B!(tmp, greens, chkr_hop_half_plus[i])
        greens .= tmp
      end
      for i in 1:n_groups
        A_mul_B!(tmp, chkr_hop_half_minus[i], greens)
        greens .= tmp
      end
  end
  nothing
end
function effective_greens2greens!(mc::DQMC_CBFalse, greens::AbstractMatrix)
  const eTminus = mc.l.hopping_matrix_exp
  const eTplus = mc.l.hopping_matrix_exp_inv
  const tmp = mc.s.tmp

  A_mul_B!(tmp, greens, eTminus)
  A_mul_B!(greens, eTplus, tmp)
  nothing
end
function effective_greens2greens!(mc::DQMC_CBFalse, U::AbstractMatrix, T::AbstractMatrix)
  const eTminus = mc.l.hopping_matrix_exp
  const eTplus = mc.l.hopping_matrix_exp_inv

  T .= T * eTminus
  U .= eTplus * U
  nothing
end
function greens2effective_greens!(mc::DQMC_CBFalse, greens::AbstractMatrix)
  const eTminus = mc.l.hopping_matrix_exp
  const eTplus = mc.l.hopping_matrix_exp_inv
  const tmp = mc.s.tmp

  A_mul_B!(tmp, greens, eTplus)
  A_mul_B!(greens, eTminus, tmp)
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
function calc_tdgf_naive(mc::AbstractDQMC, slice::Int, safe_mult::Int=mc.p.safe_mult)
  U,D,T = calc_greens_udt(mc, 1, safe_mult)

  # effective -> actual
  effective_greens2greens!(mc, U, T)

  # time displace
  Ul, Dl, Tl = calc_Bchain(mc, 1, slice, safe_mult)
  U, D, T = multiply_safely(Ul, Dl, Tl, U, D, T)

  scale!(U, D)
  return U*T
end

# Calculate "G(tau, 0)", i.e. G(slice,1) as G(slice,1) = [B(slice, 1)^-1 + B(beta, slice)]^-1 which is equal to B(slice,1)G(1)
function calc_tdgf(mc::AbstractDQMC, slice::Int, safe_mult::Int=mc.p.safe_mult)
  Ul, Dl, Tl = calc_Bchain_inv(mc, 1, slice, safe_mult)
  Ur, Dr, Tr = calc_Bchain(mc, slice, mc.p.slices, safe_mult)

  # time displace
  U, D, T = inv_sum_udts(Ul, Dl, Tl, Ur, Dr, Tr)
  effective_greens2greens!(mc, U, T)

  scale!(U, D)
  return U*T
end













## Backup before dagger stuff
# function calc_tdgf_B_udvs(mc::AbstractDQMC; inv::Bool=false, dir::String="left")
#   const G = geltype(mc)
#   const flv = mc.p.flv
#   const N = mc.l.sites
#   const nranges= length(mc.s.ranges)
#   const curr_U = mc.s.curr_U
#   const eye_full = mc.s.eye_full
#   const ones_vec = mc.s.ones_vec
#   const ranges = mc.s.ranges
  
#   u_stack = [zeros(G, flv*N, flv*N) for _ in 1:nranges]
#   d_stack = [zeros(Float64, flv*N) for _ in 1:nranges]
#   t_stack = [zeros(G, flv*N, flv*N) for _ in 1:nranges]

#   rng = 1:length(ranges)
#   dir == "right" && (rng = reverse(rng))

#   # Calculate BT1[i], given BT1[i-1]
#   @inbounds for (i, rngidx) in enumerate(rng)

#     if i == 1
#       copy!(curr_U, eye_full)
#     else
#       copy!(curr_U, u_stack[i-1])
#     end

#     slice_range = dir == "right" ? reverse(ranges[rngidx]) : ranges[rngidx]

#     for slice in slice_range
#       if inv == false
#         if dir == "left"
#           multiply_B_left!(mc, slice, curr_U)
#         else
#           warn("not working because of right multiplication!")
#           multiply_B_right!(mc, slice, curr_U)
#         end
#       else
#         if dir == "left"
#           warn("not working because of right multiplication!")
#           multiply_B_inv_right!(mc, slice, curr_U)
#         else
#           multiply_B_inv_left!(mc, slice, curr_U)
#         end
#       end
#     end

#     if i != 1
#       scale!(curr_U, d_stack[i-1])
#     end

#     u_stack[i], T = decompose_udt!(curr_U, d_stack[i])

#     if i == 1
#       A_mul_B!(t_stack[i], T, eye_full)
#     else
#       A_mul_B!(t_stack[i], T, t_stack[i-1])
#     end
#   end

#   return u_stack, d_stack, t_stack
# end













"""
Calculate UDVs at safe_mult time slices of
dir = "left": 
inv=false:  B(tau, 1) = B(tau) * B(tau-1) * ... * B(1)                    # mult left, 1:tau
inv=true:   [B(tau, 1)]^-1 = B(1)^-1 * B(2)^-1 * ... B(tau)^-1            # mult inv right, 1:tau

dir = "right":
inv=false:  B(beta, tau) = B(beta) * B(beta-1) * ... * B(tau)             # mult right, beta:tau
inv=true:   [B(beta, tau)]^-1 = B(tau)^-1 * B(tau+1)^-1 * ... B(beta)^-1  # mult inv left, beta:tau
"""
function calc_tdgf_B_udvs(mc::AbstractDQMC; inv::Bool=false, dir::String="left")
  const G = geltype(mc)
  const flv = mc.p.flv
  const N = mc.l.sites
  const nranges= length(mc.s.ranges)
  const curr_U_or_T = mc.s.curr_U
  const eye_full = mc.s.eye_full
  const ones_vec = mc.s.ones_vec
  const ranges = mc.s.ranges
  
  u_stack = [zeros(G, flv*N, flv*N) for _ in 1:nranges]
  d_stack = [zeros(Float64, flv*N) for _ in 1:nranges]
  t_stack = [zeros(G, flv*N, flv*N) for _ in 1:nranges]

  rightmult = false
  ((dir == "right" && !inv) || (dir == "left" && inv)) && (rightmult = true)

  @show rightmult

  rng = 1:length(ranges)
  dir == "right" && (rng = reverse(rng))

  # Calculate udv[i], given udv[i-1]
  @inbounds for (i, rngidx) in enumerate(rng)

    if i == 1
      copy!(curr_U_or_T, eye_full)
    else
      if !rightmult
        copy!(curr_U_or_T, u_stack[i-1])
      else
        copy!(curr_U_or_T, t_stack[i-1])
      end
    end

    slice_range = dir == "right" ? reverse(ranges[rngidx]) : ranges[rngidx]

    for slice in slice_range
      if inv == false
        if dir == "left"
          multiply_B_left!(mc, slice, curr_U_or_T)
        else
          # rightmult
          multiply_B_right!(mc, slice, curr_U_or_T)
        end
      else
        if dir == "left"
          # rightmult
          multiply_B_inv_right!(mc, slice, curr_U_or_T)
        else
          multiply_B_inv_left!(mc, slice, curr_U_or_T)
        end
      end
    end

    if i != 1
      if !rightmult
        scale!(curr_U_or_T, d_stack[i-1])
      else
        scale!(d_stack[i-1], curr_U_or_T)
      end
    end

    if !rightmult
      u_stack[i], T = decompose_udt!(curr_U_or_T, d_stack[i])
    else
      U, t_stack[i] = decompose_udt!(curr_U_or_T, d_stack[i])
    end

    if i == 1
      if !rightmult
        A_mul_B!(t_stack[i], T, eye_full)
      else
        A_mul_B!(u_stack[i], eye_full, U)
      end
    else
      if !rightmult
        A_mul_B!(t_stack[i], T, t_stack[i-1])
      else
        A_mul_B!(u_stack[i], u_stack[i-1], U)
      end
    end
  end

  return u_stack, d_stack, t_stack
end

function calc_tdgfs(mc)
  # right mult
  BT0Inv_u_stack, BT0Inv_d_stack, BT0Inv_t_stack = calc_tdgf_B_udvs(mc, inv=true, dir="left");
  BBetaT_u_stack, BBetaT_d_stack, BBetaT_t_stack = calc_tdgf_B_udvs(mc, inv=false, dir="right");
  
  # left mult
  BT0_u_stack, BT0_d_stack, BT0_t_stack = calc_tdgf_B_udvs(mc, inv=false, dir="left");
  BBetaTInv_u_stack, BBetaTInv_d_stack, BBetaTInv_t_stack = calc_tdgf_B_udvs(mc, inv=true, dir="right");
end


function test_stacks()
  # test left multiplications
  B2 = BT0_u_stack[3] * spdiagm(BT0_d_stack[3]) * BT0_t_stack[3];
  U, D, T = calc_Bchain(mc, 1, mc.s.ranges[3][end]); B1 = U*spdiagm(D)*T;
  compare(B1, B2)

  B2 = BBetaTInv_u_stack[3] * spdiagm(BBetaTInv_d_stack[3]) * BBetaTInv_t_stack[3];
  U, D, T = calc_Bchain_inv(mc, mc.s.ranges[end-2][1], mc.p.slices); B1 = U*spdiagm(D)*T;
  compare(B1, B2)




  # test right multiplications
  B2 = BT0Inv_u_stack[3] * spdiagm(BT0Inv_d_stack[3]) * BT0Inv_t_stack[3];
  U, D, T = calc_Bchain_inv(mc, 1, mc.s.ranges[3][end]); B1 = U*spdiagm(D)*T;
  compare(B1, B2)

  B2 = BBetaT_u_stack[3] * spdiagm(BBetaT_d_stack[3]) * BBetaT_t_stack[3];
  U, D, T = calc_Bchain(mc, mc.s.ranges[end-2][1], mc.p.slices); B1 = U*spdiagm(D)*T;
  compare(B1, B2)


  check_unitarity(BT0_u_stack)
  check_unitarity(BBetaTInv_u_stack)
  check_unitarity(BT0Inv_u_stack)
  check_unitarity(BBetaT_u_stack)


  # compare B(beta, 1), build by combining BBetaT and BT0
  nr = length(mc.s.ranges)
  i = floor(Int, nr/2)
  betatidx = nr - i

  U = BBetaT_t_stack[betatidx] * BT0_u_stack[i]
  scale!(U, BT0_d_stack[i])
  scale!(BBetaT_d_stack[betatidx], U)
  u,d,t = decompose_udt(U)
  u = BBetaT_u_stack[betatidx] * u
  t = t * BT0_t_stack[i]
  Bfull = u * spdiagm(d) * t
  Bfull_d = copy(d)
  U, D, T = calc_Bchain(mc, 1, mc.p.slices); Bfull2 = U * spdiagm(D) * T;
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
    !isapprox(U * ctranspose(U), eye(U)) && (return false)
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
#   scale!(D1, m1)
#   m2 = ctranspose(U1) * U2
#   scale!(m2, D2)

#   u,d,t = decompose_udt(m1+m2)

#   A = inv(t*T2)
#   B = 1./d
#   C = ctranspose(U1*u)

#   scale!(B, C)
#   return A*C
# end