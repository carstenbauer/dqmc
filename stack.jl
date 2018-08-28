mutable struct MeasStack{G<:Number} # G = GreensEltype

  # ETPCs
  etpc_evs::Array{G, 3} # j, y, x; where j ∈ ((xd,xu,xu,xd), (xd,xu,yu,yd), (yd,yu,xu,xd), (yd,yu,yu,yd))
  P::Array{Float64, 2} # ETPC

  MeasStack{G}() where G = new{G}()
end

mutable struct Stack{G<:Number} # G = GreensEltype
  u_stack::Array{G, 3}
  d_stack::Matrix{Float64}
  t_stack::Array{G, 3}

  Ul::Matrix{G}
  Ur::Matrix{G}
  Dl::Vector{Float64}
  Dr::Vector{Float64}
  Tl::Matrix{G}
  Tr::Matrix{G}

  greens::Matrix{G}
  log_det::Float64 # contains logdet of greens_{p.slices+1} === greens_1
                            # after we calculated a fresh greens in propagate()

  delta_i::Matrix{G}
  M::Matrix{G}

  ranges::Array{UnitRange, 1}
  n_elements::Int
  current_slice::Int # running internally over 0:p.slices+1, where 0 and p.slices+1 are artifcial to prepare next sweep direction.
  direction::Int

  # -------- Global update backup
  gb_u_stack::Array{G, 3}
  gb_d_stack::Matrix{Float64}
  gb_t_stack::Array{G, 3}

  gb_greens::Matrix{G}
  gb_log_det::Float64

  gb_hsfield::Array{Float64, 3}
  # --------


  #### Array allocations
  curr_U::Matrix{G}
  eV::SparseMatrixCSC{G,Int64}
  eVop1::Matrix{G}
  eVop2::Matrix{G}
  eye_flv::Matrix{Float64}
  eye_full::Matrix{Float64}
  ones_vec::Vector{Float64}

  U::Matrix{G}
  D::Vector{Float64}
  T::Matrix{G}
  # u::Matrix{G}
  d::Vector{Float64}
  # t::Matrix{G}

  greens_temp::Matrix{G}

  A::Matrix{G}
  B::Matrix{G}
  AB::Matrix{G}
  eVop1eVop2::Matrix{G}
  Mtmp::Matrix{G}
  Mtmp2::Matrix{G}
  tmp::Matrix{G}
  Bl::Matrix{G}
  tmp2::Matrix{G}
  C::Vector{G}
  S::Vector{G}
  R::Vector{G}


  #### Allocations for measurements (won't be initialized for dqmc)
  #TODO Move to meas stack!
  # TDGF
  BT0Inv_u_stack::Vector{Matrix{G}}
  BT0Inv_d_stack::Vector{Vector{Float64}}
  BT0Inv_t_stack::Vector{Matrix{G}}
  BBetaT_u_stack::Vector{Matrix{G}}
  BBetaT_d_stack::Vector{Vector{Float64}}
  BBetaT_t_stack::Vector{Matrix{G}}
  BT0_u_stack::Vector{Matrix{G}}
  BT0_d_stack::Vector{Vector{Float64}}
  BT0_t_stack::Vector{Matrix{G}}
  BBetaTInv_u_stack::Vector{Matrix{G}}
  BBetaTInv_d_stack::Vector{Vector{Float64}}
  BBetaTInv_t_stack::Vector{Matrix{G}}
  Gt0::Vector{Matrix{G}}
  G0t::Vector{Matrix{G}}

  meas::MeasStack{G}


  Stack{G}() where G = new{G}()
end


function initialize_stack(mc::AbstractDQMC)
  const s = mc.s
  const safe_mult = mc.p.safe_mult
  const N = mc.l.sites
  const flv = mc.p.flv
  const G = geltype(mc)
  const a = mc.a

  @mytimeit a.to "initialize_stack" begin

  s.n_elements = convert(Int, mc.p.slices / safe_mult) + 1

  s.eye_flv = eye(flv,flv)
  s.eye_full = eye(flv*N,flv*N)
  s.ones_vec = ones(flv*N)

  s.u_stack = zeros(G, flv*N, flv*N, s.n_elements)
  s.d_stack = zeros(Float64, flv*N, s.n_elements)
  s.t_stack = zeros(G, flv*N, flv*N, s.n_elements)

  s.greens = zeros(G, flv*N, flv*N)

  s.Ul = eye(G, flv*N, flv*N)
  s.Ur = eye(G, flv*N, flv*N)
  s.Tl = eye(G, flv*N, flv*N)
  s.Tr = eye(G, flv*N, flv*N)
  s.Dl = ones(Float64, flv*N)
  s.Dr = ones(Float64, flv*N)

  s.delta_i = zeros(G, flv, flv)
  s.M = zeros(G, flv, flv)

  # Global update backup
  s.gb_u_stack = zero(s.u_stack)
  s.gb_d_stack = zero(s.d_stack)
  s.gb_t_stack = zero(s.t_stack)
  s.gb_greens = zero(s.greens)
  s.gb_log_det = 0. 
  s.gb_hsfield = zero(mc.p.hsfield)

  s.ranges = UnitRange[]

  for i in 1:s.n_elements - 1
    push!(s.ranges, 1 + (i - 1) * safe_mult:i * safe_mult)
  end

  s.curr_U = zeros(G, flv*N, flv*N)
  s.eV = spzeros(G, flv*N, flv*N)
  s.eVop1 = zeros(G, flv, flv)
  s.eVop2 = zeros(G, flv, flv)

  # non-allocating multiplications
  # calculate_greens
  # s.U = zeros(G, flv*N, flv*N)
  s.D = zeros(Float64, flv*N)
  # s.T = zeros(G, flv*N, flv*N)
  # s.u = zeros(G, flv*N, flv*N)
  s.d = zeros(Float64, flv*N)
  # s.t = zeros(G, flv*N, flv*N)
  ## update_greens
  s.A = s.greens[:,1:N:end]
  s.B = s.greens[1:N:end,:]
  s.AB = s.A * s.B
  ## calc_detratio
  s.Mtmp = s.eye_flv - s.greens[1:N:end,1:N:end]
  s.delta_i = zeros(G, size(s.eye_flv))
  s.Mtmp2 = zeros(G, size(s.eye_flv))
  s.eVop1eVop2 = zeros(G, size(s.eye_flv))
  ## multiply_B
  s.tmp = zeros(G, flv*N, flv*N)
  s.Bl = zeros(G, flv*N, flv*N)
  ## calculate_greens
  s.tmp2 = zeros(G, flv*N, flv*N)

  s.C = zeros(G, N)
  s.S = zeros(G, N)
  s.R = zeros(G, N)

  ## propagate
  s.greens_temp = zeros(G, flv*N, flv*N)

  ## MeasStack
  s.meas = MeasStack{G}()

  end #timeit

  nothing
end


function build_stack(mc::AbstractDQMC)
  const p = mc.p
  const s = mc.s
  const a = mc.a

  @mytimeit a.to "build_stack" begin

  s.u_stack[:, :, 1] = s.eye_full
  s.d_stack[:, 1] = s.ones_vec
  s.t_stack[:, :, 1] = s.eye_full

  @inbounds for i in 1:length(s.ranges)
    add_slice_sequence_left(mc, i)
  end

  s.current_slice = p.slices + 1
  s.direction = -1

  end #timeit

  nothing
end


"""
Updates stack[idx+1] based on stack[idx]
"""
function add_slice_sequence_left(mc::AbstractDQMC, idx::Int)
  const s = mc.s
  const a = mc.a

  copy!(s.curr_U, s.u_stack[:, :, idx])

  # println("Adding slice seq left $idx = ", s.ranges[idx])
  for slice in s.ranges[idx]
    multiply_B_left!(mc, slice, s.curr_U)
  end

  @views scale!(s.curr_U, s.d_stack[:, idx])
  @mytimeit a.to "decompose_udt!" s.u_stack[:, :, idx + 1], T = @views decompose_udt!(s.curr_U, s.d_stack[:, idx + 1])
  @views A_mul_B!(s.t_stack[:, :, idx + 1],  T, s.t_stack[:, :, idx])
  nothing
end


"""
Updates stack[idx] based on stack[idx+1]
"""
function add_slice_sequence_right(mc::AbstractDQMC, idx::Int)
  const s = mc.s
  const a = mc.a

  copy!(s.curr_U, s.u_stack[:, :, idx + 1])

  for slice in reverse(s.ranges[idx])
    multiply_daggered_B_left!(mc, slice, s.curr_U)
  end

  @views scale!(s.curr_U, s.d_stack[:, idx + 1])
  @mytimeit a.to "decompose_udt!" s.u_stack[:, :, idx], T = @views decompose_udt!(s.curr_U, s.d_stack[:, idx])
  @views A_mul_B!(s.t_stack[:, :, idx], T, s.t_stack[:, :, idx + 1])
  nothing
end


function wrap_greens!(mc::AbstractDQMC, gf::Matrix, curr_slice::Int=mc.s.current_slice, direction::Int=mc.s.direction)
  if direction == -1
    multiply_B_inv_left!(mc, curr_slice - 1, gf)
    multiply_B_right!(mc, curr_slice - 1, gf)
  else
    multiply_B_left!(mc, curr_slice, gf)
    multiply_B_inv_right!(mc, curr_slice, gf)
  end
  nothing
end


function wrap_greens(mc::AbstractDQMC, gf::Matrix,slice::Int=mc.s.current_slice,direction::Int=mc.s.direction)
  temp = copy(gf)
  wrap_greens!(mc, temp, slice, direction)
  return temp
end


"""
Calculates G(slice) using s.Ur,s.Dr,s.Tr=B(slice)' ... B(M)' and s.Ul,s.Dl,s.Tl=B(slice-1) ... B(1)
"""
function calculate_greens(mc::AbstractDQMC)
  const s = mc.s
  const tmp = mc.s.tmp
  const tmp2 = mc.s.tmp2
  const a = mc.a
  @mytimeit a.to "calculate_greens" begin

  A_mul_Bc!(tmp, s.Tl, s.Tr)
  scale!(tmp, s.Dr)
  scale!(s.Dl, tmp)
  @mytimeit a.to "decompose_udt!" s.U, s.T = decompose_udt!(tmp, s.D)

  A_mul_B!(tmp, s.Ul, s.U)
  s.U .= tmp
  A_mul_Bc!(tmp2, s.T, s.Ur)
  s.T .= tmp2
  Ac_mul_B!(tmp, s.U, inv(s.T))
  tmp[diagind(tmp)] .+= s.D
  @mytimeit a.to "decompose_udt!" u, t = decompose_udt!(tmp, s.d)

  A_mul_B!(tmp, t, s.T)
  s.T = inv(tmp)
  A_mul_B!(tmp, s.U, u)
  s.U = ctranspose(tmp)
  s.d .= 1./s.d

  copy!(tmp2, s.U)
  scale!(s.d, tmp2)
  A_mul_B!(s.greens, s.T, tmp2)
  end #timeit
  nothing
end

# function calculate_greens_old(mc::AbstractDQMC)
#   const s = mc.s

#   tmp = s.Tl * ctranspose(s.Tr)
#   s.U, s.D, s.T = decompose_udt(spdiagm(s.Dl) * tmp * spdiagm(s.Dr))
#   s.U = s.Ul * s.U
#   s.T *= ctranspose(s.Ur)

#   u, s.d, t = decompose_udt(ctranspose(s.U) * inv(s.T) + spdiagm(s.D))

#   s.T = inv(t * s.T)
#   s.U *= u
#   s.U = ctranspose(s.U)
#   s.d = 1./s.d

#   s.greens = s.T * spdiagm(s.d) * s.U
#   nothing
# end


"""
Only reasonable immediately after calculate_greens() because it depends on s.U, s.d and s.T!
"""
function calculate_logdet(mc::AbstractDQMC)
  const s = mc.s

  if mc.p.opdim == 1
    s.log_det = real(log(complex(det(s.U))) + sum(log.(s.d)) + log(complex(det(s.T))))
  else
    s.log_det = real(logdet(s.U) + sum(log.(s.d)) + logdet(s.T))
  end
end


################################################################################
# Propagation
################################################################################
function propagate(mc::AbstractDQMC)
  const s = mc.s
  const p = mc.p

  if s.direction == 1
    if mod(s.current_slice, p.safe_mult) == 0
      s.current_slice +=1 # slice we are going to
      # println("we are going to $(s.current_slice) with a fresh gf.")
      if s.current_slice == 1
        s.Ur[:, :], s.Dr[:], s.Tr[:, :] = s.u_stack[:, :, 1], s.d_stack[:, 1], s.t_stack[:, :, 1]
        s.u_stack[:, :, 1] = s.eye_full
        s.d_stack[:, 1] = s.ones_vec
        s.t_stack[:, :, 1] = s.eye_full
        s.Ul[:,:], s.Dl[:], s.Tl[:,:] = s.u_stack[:, :, 1], s.d_stack[:, 1], s.t_stack[:, :, 1]

        calculate_greens(mc) # greens_1 ( === greens_{m+1} )
        calculate_logdet(mc)

      elseif 1 < s.current_slice <= p.slices
        idx = Int((s.current_slice - 1)/p.safe_mult)

        s.Ur[:, :], s.Dr[:], s.Tr[:, :] = s.u_stack[:, :, idx+1], s.d_stack[:, idx+1], s.t_stack[:, :, idx+1]
        add_slice_sequence_left(mc, idx)
        s.Ul[:,:], s.Dl[:], s.Tl[:,:] = s.u_stack[:, :, idx+1], s.d_stack[:, idx+1], s.t_stack[:, :, idx+1]

        if p.all_checks
          copy!(s.greens_temp, s.greens)
        end

        wrap_greens!(mc, s.greens_temp, s.current_slice - 1, 1)

        calculate_greens(mc) # greens_{slice we are propagating to}

        if p.all_checks
          diff = maximum(absdiff(s.greens_temp, s.greens))
          if diff > 1e-7
            @printf("->%d \t+1 Propagation instability\t %.4f\n", s.current_slice, diff)
          end
        end

      else # we are going to p.slices+1
        idx = s.n_elements - 1
        add_slice_sequence_left(mc, idx)
        s.direction = -1
        s.current_slice = p.slices+1 # redundant
        propagate(mc)
      end

    else
      # Wrapping
      wrap_greens!(mc, s.greens, s.current_slice, 1)

      s.current_slice += 1
    end

  else # s.direction == -1
    if mod(s.current_slice-1, p.safe_mult) == 0
      s.current_slice -= 1 # slice we are going to
      # println("we are going to $(s.current_slice) with a fresh+wrapped gf.")
      if s.current_slice == p.slices
        s.Ul[:, :], s.Dl[:], s.Tl[:, :] = s.u_stack[:, :, end], s.d_stack[:, end], s.t_stack[:, :, end]
        s.u_stack[:, :, end] = s.eye_full
        s.d_stack[:, end] = s.ones_vec
        s.t_stack[:, :, end] = s.eye_full
        s.Ur[:,:], s.Dr[:], s.Tr[:,:] = s.u_stack[:, :, end], s.d_stack[:, end], s.t_stack[:, :, end]

        calculate_greens(mc) # greens_{p.slices+1} === greens_1
        calculate_logdet(mc) # calculate logdet for potential global update

        # wrap to greens_{p.slices}
        wrap_greens!(mc, s.greens, s.current_slice + 1, -1)

      elseif 0 < s.current_slice < p.slices
        idx = Int(s.current_slice / p.safe_mult) + 1
        s.Ul[:, :], s.Dl[:], s.Tl[:, :] = s.u_stack[:, :, idx], s.d_stack[:, idx], s.t_stack[:, :, idx]
        add_slice_sequence_right(mc, idx)
        s.Ur[:,:], s.Dr[:], s.Tr[:,:] = s.u_stack[:, :, idx], s.d_stack[:, idx], s.t_stack[:, :, idx]

        if p.all_checks
          copy!(s.greens_temp, s.greens)
        end

        calculate_greens(mc)

        if p.all_checks
          diff = maximum(absdiff(s.greens_temp, s.greens))
          if diff > 1e-7
            @printf("->%d \t-1 Propagation instability\t %.4f\n", s.current_slice, diff)
          end
        end

        wrap_greens!(mc, s.greens, s.current_slice + 1, -1)

      else # we are going to 0
        idx = 1
        add_slice_sequence_right(mc, idx)
        s.direction = 1
        s.current_slice = 0 # redundant
        propagate(mc)
      end

    else
      # Wrapping
      wrap_greens!(mc, s.greens, s.current_slice, -1)
      s.current_slice -= 1
    end
  end
  # compare(s.greens,calculate_greens_udv(p,l,s.current_slice))
  # compare(s.greens,calculate_greens_udv_chkr(p,l,s.current_slice))
  nothing
end