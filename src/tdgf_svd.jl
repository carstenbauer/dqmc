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





# function test_Gt0()
#   eye_full = mc.s.eye_full
#   ones_vec = mc.s.ones_vec

#   Gt0 = zeros(G, Nflv, Nflv)

#   # i = 3
#   # i = 11 # == 101, almost beta half = 100
#   i = 1

#   if i != 1
#     U,D,T = inv_sum_udts(BT0Inv_u_stack[i-1], BT0Inv_d_stack[i-1], BT0Inv_t_stack[i-1],
#                  BBetaT_u_stack[i], BBetaT_d_stack[i], BBetaT_t_stack[i])
#     UDT_to_mat!(Gt0, U, D, T) # G(i,0) = G(mc.s.ranges[i][1], 0), i.e. G(21, 1) for i = 3
#     effective_greens2greens!(mc, Gt0)
#   else
#     U,D,T = inv_sum_udts(eye_full, ones_vec, eye_full,
#                  BBetaT_u_stack[i], BBetaT_d_stack[i], BBetaT_t_stack[i])
#     UDT_to_mat!(Gt0, U, D, T) # G(i,0) = G(mc.s.ranges[i][1], 0), i.e. G(21, 1) for i = 3
#     effective_greens2greens!(mc, Gt0)
#   end

#   tdgf = calc_tdgf(mc, mc.s.ranges[i][1]);
#   compare(tdgf, Gt0)

#   # compare G(0,0) with G(0) for i=1
#   g = calc_greens(mc, 1);
#   compare(Gt0, g) # 1e-16 for i=1


#   # test all Gt0 at safe mult slices
#   safe_mult_taus = 1:safe_mult:mc.p.slices
#   for tau in safe_mult_taus
#     i = ceil(Int, tau/safe_mult)
#     tdgf = calc_tdgf(mc, mc.s.ranges[i][1]);
#     if !isapprox(tdgf, Gt0[tau])
#       @show tau
#       @show i
#       break
#     end
#   end
#   # worked!

#   # TODO: test all G0t at safe mult slices
# end





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