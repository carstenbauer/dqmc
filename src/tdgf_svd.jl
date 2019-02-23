# Calculate "G(tau, 0)", i.e. G(slice,1) as
#
# G(slice,1) = [B(slice, 1)^-1 + B(beta, slice)]^-1
#
# which is (in principle) equal to B(slice,1)G(1)
function calc_tdgf_udv(mc::AbstractDQMC, slice::Int, safe_mult::Int=mc.p.safe_mult)
  if slice != 1
    Ul, Dl, Vtl = calc_Bchain_inv_udv(mc, 1, slice-1, safe_mult)
  else
    Ul, Dl, Vtl = mc.s.eye_full, mc.s.ones_vec, mc.s.eye_full
  end

  if slice != mc.p.slices
    Ur, Dr, Tr = calc_Bchain_udv(mc, slice, mc.p.slices, safe_mult)
  else
    Ur, Dr, Tr = mc.s.eye_full, mc.s.ones_vec, mc.s.eye_full
  end

  # time displace
  U, D, Vt = inv_sum_udvs(Ul, Dl, Vtl, Ur, Dr, Tr)
  effective_greens2greens!(mc, U, Vt)

  rmul!(U, Diagonal(D))
  return U*Vt
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
  curr_U_or_V = mc.s.curr_U
  eye_full = mc.s.eye_full
  ones_vec = mc.s.ones_vec
  ranges = mc.s.ranges
  
  u_stack = [zeros(G, flv*N, flv*N) for _ in 1:nranges]
  d_stack = [zeros(Float64, flv*N) for _ in 1:nranges]
  v_stack = [zeros(G, flv*N, flv*N) for _ in 1:nranges]

  rightmult = false
  ((dir == RIGHT && !inv) || (dir == LEFT && inv)) && (rightmult = true)

  # @show rightmult

  rng = 1:length(ranges)
  dir == RIGHT && (rng = reverse(rng))

  # Calculate udv[i], given udv[i-1]
  @inbounds for (i, rngidx) in enumerate(rng)

    if i == 1
      copyto!(curr_U_or_V, eye_full)
    else
      if !rightmult
        copyto!(curr_U_or_V, u_stack[i-1])
      else
        copyto!(curr_U_or_V, v_stack[i-1])
      end
    end

    slice_range = dir == RIGHT ? reverse(ranges[rngidx]) : ranges[rngidx]

    for slice in slice_range
      if inv == false
        if dir == LEFT
          multiply_B_left!(mc, slice, curr_U_or_V)
        else
          # rightmult
          multiply_B_right!(mc, slice, curr_U_or_V)
        end
      else
        if dir == LEFT
          # rightmult
          multiply_B_inv_right!(mc, slice, curr_U_or_V)
        else
          multiply_B_inv_left!(mc, slice, curr_U_or_V)
        end
      end
    end

    if i != 1
      if !rightmult
        rmul!(curr_U_or_V, Diagonal(d_stack[i-1]))
      else
        lmul!(Diagonal(d_stack[i-1]), curr_U_or_V)
      end
    end

    if !rightmult
      u_stack[i], d_stack[i], Vt = decompose_udv!(curr_U_or_V)
    else
      U, d_stack[i], v_stack[i] = decompose_udv!(curr_U_or_V)
    end

    if i == 1
      if !rightmult
        mul!(v_stack[i], Vt, eye_full)
      else
        mul!(u_stack[i], eye_full, U)
      end
    else
      if !rightmult
        mul!(v_stack[i], Vt, v_stack[i-1])
      else
        mul!(u_stack[i], u_stack[i-1], U)
      end
    end
  end

  if dir == LEFT
    return u_stack, d_stack, v_stack
  else
    return reverse(u_stack), reverse(d_stack), reverse(v_stack)
  end
end



function calc_tdgfs_udv!(mc)
  G = geltype(mc)
  M = mc.p.slices
  N = mc.l.sites
  flv = mc.p.flv
  Nflv = N * flv
  safe_mult = mc.p.safe_mult
  eye_full = mc.s.eye_full
  ones_vec = mc.s.ones_vec

  # allocate matrices if not yet done TODO: EVENTUALLY THIS SHOULD BE REMOVED
  try
    mc.s.meas.Gt0[1]
    mc.s.meas.G0t[1]
  catch
    allocate_tdgf!(mc)
  end

  Gt0 = mc.s.meas.Gt0
  G0t = mc.s.meas.G0t

  # ---- first, calculate Gt0 and G0t only at safe_mult slices 
  # right mult (Gt0)
  BT0Inv_u_stack, BT0Inv_d_stack, BT0Inv_v_stack = calc_tdgf_B_udvs(mc, inv=true, dir=LEFT);
  BBetaT_u_stack, BBetaT_d_stack, BBetaT_v_stack = calc_tdgf_B_udvs(mc, inv=false, dir=RIGHT);
  
  # left mult (G0t)
  BT0_u_stack, BT0_d_stack, BT0_v_stack = calc_tdgf_B_udvs(mc, inv=false, dir=LEFT);
  BBetaTInv_u_stack, BBetaTInv_d_stack, BBetaTInv_v_stack = calc_tdgf_B_udvs(mc, inv=true, dir=RIGHT);



  safe_mult_taus = 1:safe_mult:mc.p.slices
  @inbounds for i in 1:length(safe_mult_taus) # i = ith safe mult time slice
    tau = safe_mult_taus[i] # tau = tauth (overall) time slice
    if i != 1
      # Gt0
      inv_sum_udvs!(mc, Gt0[tau], BT0Inv_u_stack[i-1], BT0Inv_d_stack[i-1], BT0Inv_v_stack[i-1],
                   BBetaT_u_stack[i], BBetaT_d_stack[i], BBetaT_v_stack[i]) # G(i,0) = G(mc.s.ranges[i][1], 0), i.e. G(21, 1) for i = 3
      effective_greens2greens!(mc, Gt0[tau])

      # G0t
      inv_sum_udvs!(mc, G0t[tau], BT0_u_stack[i-1], BT0_d_stack[i-1], BT0_v_stack[i-1],
                   BBetaTInv_u_stack[i], BBetaTInv_d_stack[i], BBetaTInv_v_stack[i]) # G(i,0) = G(mc.s.ranges[i][1], 0), i.e. G(21, 1) for i = 3
      effective_greens2greens!(mc, G0t[tau])
    else
      # Gt0
      inv_one_plus_udv_scalettar!(mc, Gt0[tau], BBetaT_u_stack[1], BBetaT_d_stack[1], BBetaT_v_stack[1])
      effective_greens2greens!(mc, Gt0[tau])

      # G0t
      inv_one_plus_udv_scalettar!(mc, G0t[tau], BBetaTInv_u_stack[1], BBetaTInv_d_stack[1], BBetaTInv_v_stack[1])
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