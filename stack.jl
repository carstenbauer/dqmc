if !isdefined(:GreensType)
  global const GreensType = Complex128; # always complex for O(3)
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

  eye_flv::Matrix{GreensType}
  eye_full::Matrix{GreensType}
  ones_vec::Vector{Float64}

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

  Stack() = new()
end


function initialize_stack(s::Stack, p::Parameters, l::Lattice)
  s.n_elements = convert(Int, p.slices / p.safe_mult) + 1

  s.u_stack = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites, s.n_elements)
  s.d_stack = zeros(Float64, p.flv*l.sites, s.n_elements)
  s.t_stack = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites, s.n_elements)

  s.greens = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.greens_temp = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)

  s.Ul = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.Ur = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.Tl = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.Tr = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.Dl = ones(Float64, p.flv*l.sites)
  s.Dr = ones(Float64, p.flv*l.sites)

  s.U = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.D = zeros(Float64, p.flv*l.sites)
  s.T = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.u = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.d = zeros(Float64, p.flv*l.sites)
  s.t = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)

  s.delta_i = zeros(Complex{Float64}, p.flv, p.flv)
  s.M = zeros(Complex{Float64}, p.flv, p.flv)

  s.eye_flv = eye(p.flv,p.flv)
  s.eye_full = eye(p.flv*l.sites,p.flv*l.sites)

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

end


function build_stack(s::Stack, p::Parameters, l::Lattice)
  s.u_stack[:, :, 1] = s.eye_full # TODO CHECK! Is this safe?
  s.d_stack[:, 1] = ones(p.flv*l.sites)
  s.t_stack[:, :, 1] = s.eye_full

  # s.u_stack[:, :, 1] = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  # s.t_stack[:, :, 1] = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)

  # @inbounds for i in 1:length(s.ranges)
  #   add_slice_sequence_left(s, p, l, i)
  # end

  s.current_slice = p.slices + 1
  s.direction = -1

  nothing
end


"""
Updates stack[idx+1] based on stack[idx]
"""
function add_slice_sequence_left(s::Stack, p::Parameters, l::Lattice, idx::Int)
  curr_U = copy(s.u_stack[:, :, idx])
  # println("Adding slice seq left $idx = ", s.ranges[idx])
  for slice in s.ranges[idx]
    if p.chkr
      multiply_slice_matrix_left!(p, l, slice, curr_U)
    else
      curr_U = slice_matrix_no_chkr(p, l, slice) * curr_U
    end
  end

  curr_U =  curr_U * spdiagm(s.d_stack[:, idx])
  s.u_stack[:, :, idx + 1], s.d_stack[:, idx + 1], T = decompose_udt(curr_U)
  s.t_stack[:, :, idx + 1] =  T * s.t_stack[:, :, idx]
end


"""
Updates stack[idx] based on stack[idx+1]
"""
function add_slice_sequence_right(s::Stack, p::Parameters, l::Lattice, idx::Int)
  curr_U = copy(s.u_stack[:, :, idx + 1])

  for slice in reverse(s.ranges[idx])
    if p.chkr
      multiply_daggered_slice_matrix_left!(p, l, slice, curr_U)
    else
      curr_U = ctranspose(slice_matrix_no_chkr(p, l, slice)) * curr_U
    end
  end
  curr_U =  curr_U * spdiagm(s.d_stack[:, idx + 1])
  s.u_stack[:, :, idx], s.d_stack[:, idx], T = decompose_udt(curr_U)
  s.t_stack[:, :, idx] = T * s.t_stack[:, :, idx + 1]
end


# Beff(slice) = exp(−1/2∆τT)exp(−1/2∆τT)exp(−∆τV(slice))
function slice_matrix_no_chkr(p::Parameters, l::Lattice, slice::Int, power::Float64=1.)
  if power > 0
    return l.hopping_matrix_exp * l.hopping_matrix_exp * interaction_matrix_exp(p, l, slice, power)
  else
    return interaction_matrix_exp(p, l, slice, power) * l.hopping_matrix_exp_inv * l.hopping_matrix_exp_inv
  end
end


function wrap_greens_chkr!(p::Parameters, l::Lattice, gf::Matrix{GreensType}, curr_slice::Int,direction::Int)
  if direction == -1
    multiply_slice_matrix_inv_left!(p, l, curr_slice - 1, gf)
    multiply_slice_matrix_right!(p, l, curr_slice - 1, gf)
  else
    multiply_slice_matrix_left!(p, l, curr_slice, gf)
    multiply_slice_matrix_inv_right!(p, l, curr_slice, gf)
  end
end

function wrap_greens_chkr(p::Parameters, l::Lattice, gf::Matrix{GreensType},slice::Int,direction::Int)
  temp = copy(gf)
  wrap_greens_chkr!(p, l, temp, slice, direction)
  return temp
end


function wrap_greens_no_chkr!(p::Parameters, l::Lattice, gf::Matrix{GreensType}, curr_slice::Int,direction::Int)
  if direction == -1
    gf[:] = slice_matrix_no_chkr(p, l, curr_slice - 1, -1.) * gf
    gf[:] = gf * slice_matrix_no_chkr(p, l, curr_slice - 1, 1.)
  else
    gf[:] = slice_matrix_no_chkr(p, l, curr_slice, 1.) * gf
    gf[:] = gf * slice_matrix_no_chkr(p, l, curr_slice, -1.)
  end
end

function wrap_greens_no_chkr(p::Parameters, l::Lattice, gf::Matrix{GreensType},slice::Int,direction::Int)
  temp = copy(gf)
  wrap_greens_no_chkr!(p, l, temp, slice, direction)
  return temp
end


"""
Calculates G(slice) using s.Ur,s.Dr,s.Tr=B(slice)' ... B(M)' and s.Ul,s.Dl,s.Tl=B(slice-1) ... B(1)
"""
function calculate_greens(s::Stack, p::Parameters, l::Lattice)

  tmp = s.Tl * ctranspose(s.Tr)
  s.U, s.D, s.T = decompose_udt(spdiagm(s.Dl) * tmp * spdiagm(s.Dr))
  s.U = s.Ul * s.U
  s.T *= ctranspose(s.Ur)

  s.u, s.d, s.t = decompose_udt(/(ctranspose(s.U), s.T) + spdiagm(s.D))

  s.T = inv(s.t * s.T)
  s.U *= s.u
  s.U = ctranspose(s.U)
  s.d = 1./s.d

  s.greens = s.T * spdiagm(s.d) * s.U
end

"""
Only reasonable immediately after calculate_greens()!
"""
function calculate_logdet(s::Stack, p::Parameters, l::Lattice)
  s.log_det = real(logdet(s.U) + sum(log.(s.d)) + logdet(s.T))
end


################################################################################
# Propagation
################################################################################
function propagate(s::Stack, p::Parameters, l::Lattice)
  if s.direction == 1
    if mod(s.current_slice, p.safe_mult) == 0
      s.current_slice +=1 # slice we are going to
      if s.current_slice == 1
        s.Ur[:, :], s.Dr[:], s.Tr[:, :] = s.u_stack[:, :, 1], s.d_stack[:, 1], s.t_stack[:, :, 1]
        s.u_stack[:, :, 1] = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
        s.d_stack[:, 1] = ones(p.flv*l.sites)
        s.t_stack[:, :, 1] = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
        s.Ul[:,:], s.Dl[:], s.Tl[:,:] = s.u_stack[:, :, 1], s.d_stack[:, 1], s.t_stack[:, :, 1]

        calculate_greens(s, p, l) # greens_1 ( === greens_{m+1} )
        calculate_logdet(s, p, l)

      elseif 1 < s.current_slice <= p.slices
        idx = Int((s.current_slice - 1)/p.safe_mult)

        s.Ur[:, :], s.Dr[:], s.Tr[:, :] = s.u_stack[:, :, idx+1], s.d_stack[:, idx+1], s.t_stack[:, :, idx+1]
        add_slice_sequence_left(s, p, l, idx)
        s.Ul[:,:], s.Dl[:], s.Tl[:,:] = s.u_stack[:, :, idx+1], s.d_stack[:, idx+1], s.t_stack[:, :, idx+1]

        s.greens_temp = copy(s.greens)
        if p.chkr
          wrap_greens_chkr!(p, l, s.greens_temp, s.current_slice - 1, 1)
        else
          wrap_greens_no_chkr!(p, l, s.greens_temp, s.current_slice - 1, 1)
        end

        calculate_greens(s, p, l) # greens_{slice we are propagating to}

        if p.all_checks
          diff = maximum(absdiff(s.greens_temp, s.greens))
          if diff > 1e-7
            @printf("->%d \t+1 Propagation instability\t %.4f\n", s.current_slice, diff)
          end
        end

      else # we are going to p.slices+1
        idx = s.n_elements - 1
        add_slice_sequence_left(s, p, l, idx)
        s.direction = -1
        s.current_slice = p.slices+1 # redundant
        propagate(s, p, l)
      end

    else
      # Wrapping
      if p.chkr
        wrap_greens_chkr!(p, l, s.greens, s.current_slice, 1)
      else
        wrap_greens_no_chkr!(p, l, s.greens, s.current_slice, 1)
      end
      s.current_slice += 1
    end

  else # s.direction == -1
    if mod(s.current_slice-1, p.safe_mult) == 0
      s.current_slice -= 1 # slice we are going to
      if s.current_slice == p.slices
        s.Ul[:, :], s.Dl[:], s.Tl[:, :] = s.u_stack[:, :, end], s.d_stack[:, end], s.t_stack[:, :, end]
        s.u_stack[:, :, end] = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
        s.d_stack[:, end] = ones(p.flv*l.sites)
        s.t_stack[:, :, end] = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
        s.Ur[:,:], s.Dr[:], s.Tr[:,:] = s.u_stack[:, :, end], s.d_stack[:, end], s.t_stack[:, :, end]

        calculate_greens(s, p, l) # greens_{p.slices+1} === greens_1
        calculate_logdet(s, p, l) # calculate logdet for potential global update

        # wrap to greens_{p.slices}
        if p.chkr
          wrap_greens_chkr!(p, l, s.greens, s.current_slice + 1, -1)
        else
          wrap_greens_no_chkr!(p, l, s.greens, s.current_slice + 1, -1)
        end

      elseif 0 < s.current_slice < p.slices
        idx = Int(s.current_slice / p.safe_mult) + 1
        s.Ul[:, :], s.Dl[:], s.Tl[:, :] = s.u_stack[:, :, idx], s.d_stack[:, idx], s.t_stack[:, :, idx]
        add_slice_sequence_right(s, p, l, idx)
        s.Ur[:,:], s.Dr[:], s.Tr[:,:] = s.u_stack[:, :, idx], s.d_stack[:, idx], s.t_stack[:, :, idx]

        s.greens_temp = copy(s.greens)

        calculate_greens(s, p , l)

        if p.all_checks
          diff = maximum(absdiff(s.greens_temp, s.greens))
          if diff > 1e-7
            @printf("->%d \t-1 Propagation instability\t %.4f\n", s.current_slice, diff)
          end
        end

        if p.chkr
          wrap_greens_chkr!(p, l, s.greens, s.current_slice + 1, -1)
        else
          wrap_greens_no_chkr!(p, l, s.greens, s.current_slice + 1, -1)
        end

      else # we are going to 0
        idx = 1
        add_slice_sequence_right(s, p, l, idx)
        s.direction = 1
        s.current_slice = 0 # redundant
        propagate(s,p,l)
      end

    else
      # Wrapping
      if p.chkr
        wrap_greens_chkr!(p, l, s.greens, s.current_slice, -1)
      else
        wrap_greens_no_chkr!(p, l, s.greens, s.current_slice, -1)
      end
      s.current_slice -= 1
    end
  end
  # compare(s.greens,calculate_greens_udv(p,l,s.current_slice))
  # compare(s.greens,calculate_greens_udv_chkr(p,l,s.current_slice))
  nothing
end