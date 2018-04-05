if !isdefined(:GreensType)
  global const GreensType = Complex128; # assume O(2) or O(3)
  warn("GreensType wasn't set on loading stack.jl")
  println("GreensType = ", GreensType)
end

mutable struct Stack
  u_stack::Array{GreensType, 3}
  d_stack::Matrix{Float64}
  t_stack::Array{GreensType, 3}

  Ul::Matrix{GreensType}
  Ur::Matrix{GreensType}
  Dl::Vector{Float64}
  Dr::Vector{Float64}
  Tl::Matrix{GreensType}
  Tr::Matrix{GreensType}

  greens::Matrix{GreensType}
  greens_temp::Matrix{GreensType}
  log_det::Float64 # contains logdet of greens_{p.slices+1} === greens_1
                            # after we calculated a fresh greens in propagate()

  U::Matrix{GreensType}
  D::Vector{Float64}
  T::Matrix{GreensType}
  u::Matrix{GreensType}
  d::Vector{Float64}
  t::Matrix{GreensType}

  delta_i::Matrix{GreensType}
  M::Matrix{GreensType}

  ranges::Array{UnitRange, 1}
  n_elements::Int
  current_slice::Int # running internally over 0:p.slices+1, where 0 and p.slices+1 are artifcial to prepare next sweep direction.
  direction::Int

  # -------- Global update backup
  gb_u_stack::Array{GreensType, 3}
  gb_d_stack::Matrix{Float64}
  gb_t_stack::Array{GreensType, 3}

  gb_greens::Matrix{GreensType}
  gb_log_det::Float64

  gb_hsfield::Array{Float64, 3}
  # --------


  #### Array allocations
  curr_U::Matrix{GreensType}
  eV::Matrix{GreensType}
  eVop1::Matrix{GreensType}
  eVop2::Matrix{GreensType}

  Stack() = new()
end


function initialize_stack(mc::AbstractDQMC)
  const p = mc.p
  const l = mc.l
  const s = mc.s

  s.n_elements = convert(Int, p.slices / p.safe_mult) + 1

  s.u_stack = zeros(GreensType, p.flv*l.sites, p.flv*l.sites, s.n_elements)
  s.d_stack = zeros(Float64, p.flv*l.sites, s.n_elements)
  s.t_stack = zeros(GreensType, p.flv*l.sites, p.flv*l.sites, s.n_elements)

  s.greens = zeros(GreensType, p.flv*l.sites, p.flv*l.sites)
  s.greens_temp = zeros(GreensType, p.flv*l.sites, p.flv*l.sites)

  s.Ul = eye(GreensType, p.flv*l.sites, p.flv*l.sites)
  s.Ur = eye(GreensType, p.flv*l.sites, p.flv*l.sites)
  s.Tl = eye(GreensType, p.flv*l.sites, p.flv*l.sites)
  s.Tr = eye(GreensType, p.flv*l.sites, p.flv*l.sites)
  s.Dl = ones(Float64, p.flv*l.sites)
  s.Dr = ones(Float64, p.flv*l.sites)

  s.U = zeros(GreensType, p.flv*l.sites, p.flv*l.sites)
  s.D = zeros(Float64, p.flv*l.sites)
  s.T = zeros(GreensType, p.flv*l.sites, p.flv*l.sites)
  s.u = zeros(GreensType, p.flv*l.sites, p.flv*l.sites)
  s.d = zeros(Float64, p.flv*l.sites)
  s.t = zeros(GreensType, p.flv*l.sites, p.flv*l.sites)

  s.delta_i = zeros(GreensType, p.flv, p.flv)
  s.M = zeros(GreensType, p.flv, p.flv)

  # Global update backup
  s.gb_u_stack = zero(s.u_stack)
  s.gb_d_stack = zero(s.d_stack)
  s.gb_t_stack = zero(s.t_stack)
  s.gb_greens = zero(s.greens)
  s.gb_log_det = 0. 
  s.gb_hsfield = zero(p.hsfield)

  s.ranges = UnitRange[]

  for i in 1:s.n_elements - 1
    push!(s.ranges, 1 + (i - 1) * p.safe_mult:i * p.safe_mult)
  end

  s.curr_U = zero(s.U)
  s.eV = zeros(GreensType, p.flv * l.sites, p.flv * l.sites)
  s.eVop1 = zeros(GreensType, p.flv, p.flv)
  s.eVop2 = zeros(GreensType, p.flv, p.flv)

end


function build_stack(mc::AbstractDQMC)
  const p = mc.p
  const s = mc.s

  s.u_stack[:, :, 1] = eye_full
  s.d_stack[:, 1] = ones_vec
  s.t_stack[:, :, 1] = eye_full

  @inbounds for i in 1:length(s.ranges)
    add_slice_sequence_left(mc, i)
  end

  s.current_slice = p.slices + 1
  s.direction = -1

  nothing
end


"""
Updates stack[idx+1] based on stack[idx]
"""
function add_slice_sequence_left(mc::AbstractDQMC, idx::Int)
  const s = mc.s

  copy!(s.curr_U, s.u_stack[:, :, idx])

  # println("Adding slice seq left $idx = ", s.ranges[idx])
  for slice in s.ranges[idx]
    if mc.p.chkr
      multiply_slice_matrix_left!(mc, slice, s.curr_U)
    else
      s.curr_U = slice_matrix_no_chkr(mc, slice) * s.curr_U
    end
  end

  s.curr_U *= spdiagm(s.d_stack[:, idx])
  s.u_stack[:, :, idx + 1], s.d_stack[:, idx + 1], T = decompose_udt(s.curr_U)
  s.t_stack[:, :, idx + 1] =  T * s.t_stack[:, :, idx]
end


"""
Updates stack[idx] based on stack[idx+1]
"""
function add_slice_sequence_right(mc::AbstractDQMC, idx::Int)
  const s = mc.s

  copy!(s.curr_U, s.u_stack[:, :, idx + 1])

  for slice in reverse(s.ranges[idx])
    if mc.p.chkr
      multiply_daggered_slice_matrix_left!(mc, slice, s.curr_U)
    else
      s.curr_U = ctranspose(slice_matrix_no_chkr(mc, slice)) * s.curr_U
    end
  end

  s.curr_U *=  spdiagm(s.d_stack[:, idx + 1])
  s.u_stack[:, :, idx], s.d_stack[:, idx], T = decompose_udt(s.curr_U)
  s.t_stack[:, :, idx] = T * s.t_stack[:, :, idx + 1]
end


@inline function wrap_greens_chkr!(mc::AbstractDQMC{C}, gf::Matrix{GreensType}, curr_slice::Int,direction::Int) where C<:CBTrue
  if direction == -1
    multiply_slice_matrix_inv_left!(mc, curr_slice - 1, gf)
    multiply_slice_matrix_right!(mc, curr_slice - 1, gf)
  else
    multiply_slice_matrix_left!(mc, curr_slice, gf)
    multiply_slice_matrix_inv_right!(mc, curr_slice, gf)
  end
end

function wrap_greens_chkr(mc::AbstractDQMC{C}, gf::Matrix{GreensType},slice::Int,direction::Int) where C<:CBTrue
  temp = copy(gf)
  wrap_greens_chkr!(mc, temp, slice, direction)
  return temp
end


function wrap_greens_no_chkr!(mc::AbstractDQMC{C}, gf::Matrix{GreensType}, curr_slice::Int,direction::Int) where C<:CBFalse
  if direction == -1
    gf[:] = slice_matrix_no_chkr(mc, curr_slice - 1, -1.) * gf
    gf[:] = gf * slice_matrix_no_chkr(mc, curr_slice - 1, 1.)
  else
    gf[:] = slice_matrix_no_chkr(mc, curr_slice, 1.) * gf
    gf[:] = gf * slice_matrix_no_chkr(mc, curr_slice, -1.)
  end
end

function wrap_greens_no_chkr(mc::AbstractDQMC{C}, gf::Matrix{GreensType},slice::Int,direction::Int) where C<:CBFalse
  temp = copy(gf)
  wrap_greens_no_chkr!(mc, temp, slice, direction)
  return temp
end


# Beff(slice) = exp(−1/2∆τT)exp(−1/2∆τT)exp(−∆τV(slice))
function slice_matrix_no_chkr(mc::AbstractDQMC{C}, slice::Int, power::Float64=1.) where C<:CBFalse
  const eT = mc.l.hopping_matrix_exp
  const eTinv = mc.l.hopping_matrix_exp_inv
  const eV = interaction_matrix_exp(mc, slice, power)

  if power > 0
    return eT * eT * eV
  else
    return eV * eTinv * eTinv
  end
end


"""
Calculates G(slice) using s.Ur,s.Dr,s.Tr=B(slice)' ... B(M)' and s.Ul,s.Dl,s.Tl=B(slice-1) ... B(1)
"""
function calculate_greens(mc::AbstractDQMC)
  const s = mc.s

  tmp = s.Tl * ctranspose(s.Tr)
  s.U, s.D, s.T = decompose_udt(spdiagm(s.Dl) * tmp * spdiagm(s.Dr))
  s.U = s.Ul * s.U
  s.T *= ctranspose(s.Ur)

  s.u, s.d, s.t = decompose_udt(ctranspose(s.U) * inv(s.T) + spdiagm(s.D))

  s.T = inv(s.t * s.T)
  s.U *= s.u
  s.U = ctranspose(s.U)
  s.d = 1./s.d

  s.greens = s.T * spdiagm(s.d) * s.U
end


"""
Only reasonable immediately after calculate_greens()!
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
      if s.current_slice == 1
        s.Ur[:, :], s.Dr[:], s.Tr[:, :] = s.u_stack[:, :, 1], s.d_stack[:, 1], s.t_stack[:, :, 1]
        s.u_stack[:, :, 1] = eye_full
        s.d_stack[:, 1] = ones_vec
        s.t_stack[:, :, 1] = eye_full
        s.Ul[:,:], s.Dl[:], s.Tl[:,:] = s.u_stack[:, :, 1], s.d_stack[:, 1], s.t_stack[:, :, 1]

        calculate_greens(mc) # greens_1 ( === greens_{m+1} )
        calculate_logdet(mc)

      elseif 1 < s.current_slice <= p.slices
        idx = Int((s.current_slice - 1)/p.safe_mult)

        s.Ur[:, :], s.Dr[:], s.Tr[:, :] = s.u_stack[:, :, idx+1], s.d_stack[:, idx+1], s.t_stack[:, :, idx+1]
        add_slice_sequence_left(mc, idx)
        s.Ul[:,:], s.Dl[:], s.Tl[:,:] = s.u_stack[:, :, idx+1], s.d_stack[:, idx+1], s.t_stack[:, :, idx+1]

        if p.all_checks
          s.greens_temp = copy(s.greens)
        end

        if p.chkr
          wrap_greens_chkr!(mc, s.greens_temp, s.current_slice - 1, 1)
        else
          wrap_greens_no_chkr!(mc, s.greens_temp, s.current_slice - 1, 1)
        end

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
      if p.chkr
        wrap_greens_chkr!(mc, s.greens, s.current_slice, 1)
      else
        wrap_greens_no_chkr!(mc, s.greens, s.current_slice, 1)
      end
      s.current_slice += 1
    end

  else # s.direction == -1
    if mod(s.current_slice-1, p.safe_mult) == 0
      s.current_slice -= 1 # slice we are going to
      if s.current_slice == p.slices
        s.Ul[:, :], s.Dl[:], s.Tl[:, :] = s.u_stack[:, :, end], s.d_stack[:, end], s.t_stack[:, :, end]
        s.u_stack[:, :, end] = eye_full
        s.d_stack[:, end] = ones_vec
        s.t_stack[:, :, end] = eye_full
        s.Ur[:,:], s.Dr[:], s.Tr[:,:] = s.u_stack[:, :, end], s.d_stack[:, end], s.t_stack[:, :, end]

        calculate_greens(mc) # greens_{p.slices+1} === greens_1
        calculate_logdet(mc) # calculate logdet for potential global update

        # wrap to greens_{p.slices}
        if p.chkr
          wrap_greens_chkr!(mc, s.greens, s.current_slice + 1, -1)
        else
          wrap_greens_no_chkr!(mc, s.greens, s.current_slice + 1, -1)
        end

      elseif 0 < s.current_slice < p.slices
        idx = Int(s.current_slice / p.safe_mult) + 1
        s.Ul[:, :], s.Dl[:], s.Tl[:, :] = s.u_stack[:, :, idx], s.d_stack[:, idx], s.t_stack[:, :, idx]
        add_slice_sequence_right(mc, idx)
        s.Ur[:,:], s.Dr[:], s.Tr[:,:] = s.u_stack[:, :, idx], s.d_stack[:, idx], s.t_stack[:, :, idx]

        if p.all_checks
          s.greens_temp = copy(s.greens)
        end

        calculate_greens(mc)

        if p.all_checks
          diff = maximum(absdiff(s.greens_temp, s.greens))
          if diff > 1e-7
            @printf("->%d \t-1 Propagation instability\t %.4f\n", s.current_slice, diff)
          end
        end

        if p.chkr
          wrap_greens_chkr!(mc, s.greens, s.current_slice + 1, -1)
        else
          wrap_greens_no_chkr!(mc, s.greens, s.current_slice + 1, -1)
        end

      else # we are going to 0
        idx = 1
        add_slice_sequence_right(mc, idx)
        s.direction = 1
        s.current_slice = 0 # redundant
        propagate(mc)
      end

    else
      # Wrapping
      if p.chkr
        wrap_greens_chkr!(mc, s.greens, s.current_slice, -1)
      else
        wrap_greens_no_chkr!(mc, s.greens, s.current_slice, -1)
      end
      s.current_slice -= 1
    end
  end
  # compare(s.greens,calculate_greens_udv(p,l,s.current_slice))
  # compare(s.greens,calculate_greens_udv_chkr(p,l,s.current_slice))
  nothing
end