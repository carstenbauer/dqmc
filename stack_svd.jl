mutable struct Stack
  u_stack::Array{Complex{Float64}, 3}
  d_stack::Array{Float64, 2}
  vt_stack::Array{Complex{Float64}, 3}

  Ul::Array{Complex{Float64}, 2}
  Ur::Array{Complex{Float64}, 2}
  Dl::Array{Float64, 1}
  Dr::Array{Float64, 1}
  Vtl::Array{Complex{Float64}, 2}
  Vtr::Array{Complex{Float64}, 2}

  greens::Array{Complex{Float64}, 2}
  greens_inv_svs::Vector{Float64} # only valid after fresh calculation of green's function (not after wrapping/update)
  greens_temp::Array{Complex{Float64}, 2}

  U::Array{Complex{Float64}, 2}
  D::Array{Float64, 1}
  Vt::Array{Complex{Float64}, 2}

  delta_i::Array{Complex{Float64}, 2}
  M::Array{Complex{Float64}, 2}

  eye_flv::Array{Complex{Float64},2}
  eye_full::Array{Complex{Float64},2}

  ranges::Array{UnitRange, 1}
  n_elements::Int
  current_slice::Int # running internally over 0:p.slices+1, where 0 and p.slices+1 are artifcial to prepare next sweep direction.
  direction::Int

  # -------- Global update backup
  gb_u_stack::Array{Complex{Float64}, 3}
  gb_d_stack::Array{Float64, 2}
  gb_vt_stack::Array{Complex{Float64}, 3}

  gb_greens::Array{Complex{Float64}, 2}
  gb_greens_inv_svs::Vector{Float64}

  gb_hsfield::Array{Float64, 3}
  # --------

  Stack() = new()
end


function initialize_stack(s::Stack, p::Parameters, l::Lattice)
  s.n_elements = convert(Int, p.slices / p.safe_mult) + 1

  s.ranges = UnitRange[]

  for i in 1:s.n_elements - 1
    push!(s.ranges, 1 + (i - 1) * p.safe_mult:i * p.safe_mult)
  end

  s.u_stack = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites, s.n_elements)
  s.d_stack = zeros(Float64, p.flv*l.sites, s.n_elements)
  s.vt_stack = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites, s.n_elements)

  s.greens = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.greens_inv_svs = zeros(p.flv*l.sites)
  s.greens_temp = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)

  s.Ul = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.Ur = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.Vtl = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.Vtr = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.Dl = ones(Float64, p.flv*l.sites)
  s.Dr = ones(Float64, p.flv*l.sites)

  s.U = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.D = zeros(Float64, p.flv*l.sites)
  s.Vt = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)


  s.delta_i = zeros(Complex{Float64}, p.flv, p.flv)
  s.M = zeros(Complex{Float64}, p.flv, p.flv)

  eye_flv = eye(p.flv,p.flv)
  eye_full = eye(p.flv*l.sites,p.flv*l.sites)

  # Global update backup
  s.gb_u_stack = similar(s.u_stack)
  s.gb_d_stack = similar(s.d_stack)
  s.gb_vt_stack = similar(s.vt_stack)
  s.gb_greens = similar(s.greens)
  s.gb_greens_inv_svs = similar(s.greens_inv_svs)
  s.gb_hsfield = similar(p.hsfield)

  nothing
end


function build_stack(s::Stack, p::Parameters, l::Lattice)
  s.u_stack[:, :, 1] = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.d_stack[:, 1] = ones(p.flv*l.sites)
  s.vt_stack[:, :, 1] = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)

  @inbounds for i in 1:length(s.ranges)
    add_slice_sequence_left(s, p, l, i)
  end

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
    # curr_U = slice_matrix_no_chkr(p, l, slice) * curr_U
    multiply_slice_matrix_left!(p, l, slice, curr_U)
  end

  curr_U =  curr_U * spdiagm(s.d_stack[:, idx])
  s.u_stack[:, :, idx + 1], s.d_stack[:, idx + 1], s.Vt = decompose_udv!(curr_U)
  s.vt_stack[:, :, idx + 1] =  s.Vt * s.vt_stack[:, :, idx]
end


"""
Updates stack[idx] based on stack[idx+1]
"""
function add_slice_sequence_right(s::Stack, p::Parameters, l::Lattice, idx::Int)
  curr_Vt = copy(s.vt_stack[:, :, idx + 1])

  for slice in reverse(s.ranges[idx])
    # curr_Vt = curr_Vt * slice_matrix_no_chkr(p, l, slice)
    multiply_slice_matrix_right!(p, l, slice, curr_Vt)
  end
  curr_Vt =  spdiagm(s.d_stack[:, idx + 1]) * curr_Vt
  s.U, s.d_stack[:, idx], s.vt_stack[:, :, idx] = decompose_udv!(curr_Vt)
  s.u_stack[:, :, idx] = s.u_stack[:, :, idx + 1] * s.U
end

# Beff(slice) = exp(−1/2∆τT)exp(−1/2∆τT)exp(−∆τV(slice))
function slice_matrix_no_chkr(p::Parameters, l::Lattice, slice::Int, power::Float64=1.)
  if power > 0
    return l.hopping_matrix_exp * l.hopping_matrix_exp * interaction_matrix_exp(p, l, slice, power)
  else
    return interaction_matrix_exp(p, l, slice, power) * l.hopping_matrix_exp_inv * l.hopping_matrix_exp_inv
  end
end


function wrap_greens_chkr!(p::Parameters, l::Lattice, gf::Array{Complex{Float64},2}, curr_slice::Int,direction::Int)
  if direction == -1
    multiply_slice_matrix_inv_left!(p, l, curr_slice - 1, gf)
    multiply_slice_matrix_right!(p, l, curr_slice - 1, gf)
  else
    multiply_slice_matrix_left!(p, l, curr_slice, gf)
    multiply_slice_matrix_inv_right!(p, l, curr_slice, gf)
  end
end

function wrap_greens_chkr(p::Parameters, l::Lattice, gf::Array{Complex{Float64},2},slice::Int,direction::Int)
  temp = copy(gf)
  wrap_greens_chkr!(p, l, temp, slice, direction)
  return temp
end


function wrap_greens_no_chkr!(p::Parameters, l::Lattice, gf::Array{Complex{Float64},2}, curr_slice::Int,direction::Int)
  if direction == -1
    gf[:] = slice_matrix_no_chkr(p, l, curr_slice - 1, -1.) * gf
    gf[:] = gf * slice_matrix_no_chkr(p, l, curr_slice - 1, 1.)
  else
    gf[:] = slice_matrix_no_chkr(p, l, curr_slice, 1.) * gf
    gf[:] = gf * slice_matrix_no_chkr(p, l, curr_slice, -1.)
  end
end

function wrap_greens_no_chkr(p::Parameters, l::Lattice, gf::Array{Complex{Float64},2},slice::Int,direction::Int)
  temp = copy(gf)
  wrap_greens_no_chkr!(p, l, temp, slice, direction)
  return temp
end


"""
Calculates G(slice) using s.Ur,s.Dr,s.Vtr=B(M) ... B(slice) and s.Ul,s.Dl,s.Vtl=B(slice-1) ... B(1)
"""
function calculate_greens(s::Stack, p::Parameters, l::Lattice)
  tmp = s.Vtl * s.Ur
  inner = ctranspose(s.Vtr * s.Ul) + spdiagm(s.Dl) * tmp * spdiagm(s.Dr)
  s.U, s.D, s.Vt = decompose_udv!(inner)
  s.greens_inv_svs = s.D
  s.greens = ctranspose(s.Vt * s.Vtr) * spdiagm(1./s.D) * ctranspose(s.Ul * s.U)
end


################################################################################
# Propagation
################################################################################
function propagate(s::Stack, p::Parameters, l::Lattice)
  if s.direction == 1
    if mod(s.current_slice, p.safe_mult) == 0
      s.current_slice +=1 # slice we are going to
      if s.current_slice == 1
        s.Ur[:, :], s.Dr[:], s.Vtr[:, :] = s.u_stack[:, :, 1], s.d_stack[:, 1], s.vt_stack[:, :, 1]
        s.u_stack[:, :, 1] = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
        s.d_stack[:, 1] = ones(p.flv*l.sites)
        s.vt_stack[:, :, 1] = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
        s.Ul[:,:], s.Dl[:], s.Vtl[:,:] = s.u_stack[:, :, 1], s.d_stack[:, 1], s.vt_stack[:, :, 1]

        calculate_greens(s, p, l) # greens_1 ( === greens_{m+1} )

      elseif 1 < s.current_slice <= p.slices
        idx = Int((s.current_slice - 1)/p.safe_mult)

        s.Ur[:, :], s.Dr[:], s.Vtr[:, :] = s.u_stack[:, :, idx+1], s.d_stack[:, idx+1], s.vt_stack[:, :, idx+1]
        add_slice_sequence_left(s, p, l, idx)
        s.Ul[:,:], s.Dl[:], s.Vtl[:,:] = s.u_stack[:, :, idx+1], s.d_stack[:, idx+1], s.vt_stack[:, :, idx+1]

        s.greens_temp = copy(s.greens)
        wrap_greens_chkr!(p, l, s.greens_temp, s.current_slice - 1, 1)

        calculate_greens(s, p, l) # greens_{slice we are propagating to}

        errs = (effreldiff(s.greens_temp, s.greens) .> 1e-2) & (absdiff(s.greens_temp, s.greens) .> 1e-04)
        if sum(errs)>0
          maxrelerr = maximum(effreldiff(s.greens_temp, s.greens)[errs])*100
          maxabsolute = maximum(absdiff(s.greens_temp, s.greens)[errs])
          @printf("->%d \t+1 Propagation stability\t max absolute: %.4f \t max relative: %.1f%%\n", s.current_slice, maxabsolute, maxrelerr)
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
      wrap_greens_chkr!(p, l, s.greens, s.current_slice, 1)
      s.current_slice += 1
    end

  else # s.direction == -1
    if mod(s.current_slice-1, p.safe_mult) == 0
      s.current_slice -= 1 # slice we are going to
      if s.current_slice == p.slices
        s.Ul[:, :], s.Dl[:], s.Vtl[:, :] = s.u_stack[:, :, end], s.d_stack[:, end], s.vt_stack[:, :, end]
        s.u_stack[:, :, end] = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
        s.d_stack[:, end] = ones(p.flv*l.sites)
        s.vt_stack[:, :, end] = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
        s.Ur[:,:], s.Dr[:], s.Vtr[:,:] = s.u_stack[:, :, end], s.d_stack[:, end], s.vt_stack[:, :, end]

        calculate_greens(s, p, l) # greens_{p.slices+1} === greens_1

        # wrap to greens_{p.slices}
        wrap_greens_chkr!(p, l, s.greens, s.current_slice + 1, -1)

      elseif 0 < s.current_slice < p.slices
        idx = Int(s.current_slice / p.safe_mult) + 1
        s.Ul[:, :], s.Dl[:], s.Vtl[:, :] = s.u_stack[:, :, idx], s.d_stack[:, idx], s.vt_stack[:, :, idx]
        add_slice_sequence_right(s, p, l, idx)
        s.Ur[:,:], s.Dr[:], s.Vtr[:,:] = s.u_stack[:, :, idx], s.d_stack[:, idx], s.vt_stack[:, :, idx]

        s.greens_temp = copy(s.greens)

        calculate_greens(s, p , l)

        # diff = maximum(absdiff(s.greens_temp, s.greens))
        # if diff > 1e-4
        #   @printf("->%d \t-1 Propagation stability\t %.4f\n", s.current_slice, diff)
        # end

        errs = (effreldiff(s.greens_temp, s.greens) .> 1e-2) & (absdiff(s.greens_temp, s.greens) .> 1e-04)
        if sum(errs)>0
          maxrelerr = maximum(effreldiff(s.greens_temp, s.greens)[errs])*100
          maxabsolute = maximum(absdiff(s.greens_temp, s.greens)[errs])
          @printf("->%d \t-1 Propagation stability\t max absolute: %.4f \t max relative: %.1f%%\n", s.current_slice, maxabsolute, maxrelerr)
        end

        wrap_greens_chkr!(p, l, s.greens, s.current_slice + 1, -1)

      else # we are going to 0
        idx = 1
        add_slice_sequence_right(s, p, l, idx)
        s.direction = 1
        s.current_slice = 0 # redundant
        propagate(s,p,l)
      end

    else
      # Wrapping
      wrap_greens_chkr!(p, l, s.greens, s.current_slice, -1)
      s.current_slice -= 1
    end
  end
  # compare(s.greens,calculate_greens_udv(p,l,s.current_slice))
  # compare(s.greens,calculate_greens_udv_chkr(p,l,s.current_slice))
  nothing
end
