type stack
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
  greens_temp::Array{Complex{Float64}, 2}

  U::Array{Complex{Float64}, 2}
  D::Array{Float64, 1}
  V::Array{Complex{Float64}, 2}

  delta_i::Array{Complex{Float64}, 2}
  M::Array{Complex{Float64}, 2}

  ranges::Array{UnitRange, 1}
  n_elements::Int
  current_slice::Int
  direction::Int

  stack() = new()
end


function initialize_stack(s::stack, p::parameters, l::lattice)
  s.n_elements = convert(Int, p.slices / p.safe_mult) + 1

  s.u_stack = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites, s.n_elements)
  s.d_stack = zeros(Float64, p.flv*l.sites, s.n_elements)
  s.vt_stack = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites, s.n_elements)

  s.greens = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.greens_temp = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)

  s.Ul = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.Ur = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.Vtl = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.Vtr = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.Dl = ones(Float64, p.flv*l.sites)
  s.Dr = ones(Float64, p.flv*l.sites)

  s.U = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.D = zeros(Float64, p.flv*l.sites)
  s.V = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)

  s.delta_i = zeros(Complex{Float64}, p.flv, p.flv)
  s.M = zeros(Complex{Float64}, p.flv, p.flv)

  s.ranges = UnitRange[]

  for i in 1:s.n_elements - 1
    push!(s.ranges, 1 + (i - 1) * p.safe_mult:i * p.safe_mult)
  end

end


function build_stack(s::stack, p::parameters, l::lattice)
  println("Building stack")
  s.u_stack[:, :, 1] = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.d_stack[:, 1] = ones(p.flv*l.sites)
  s.vt_stack[:, :, 1] = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)

  @inbounds for i in 1:length(s.ranges)
    add_slice_sequence_left(s, p, l, i)
  end

  s.current_slice = p.slices + 1
  s.direction = -1
end


"""
Updates stack[idx+1] based on stack[idx]
"""
function add_slice_sequence_left(s::stack, p::parameters, l::lattice, idx::Int)
  curr_U = copy(s.u_stack[:, :, idx])
  # println("Adding slice seq left $idx = ", s.ranges[idx])
  for slice in s.ranges[idx]
    curr_U = slice_matrix_no_chkr(p, l, slice) * curr_U
  end

  curr_U =  curr_U * spdiagm(s.d_stack[:, idx])
  F = decompose_udv!(curr_U)
  s.u_stack[:, :, idx + 1] = F[:U]
  s.d_stack[:, idx + 1] = F[:S]
  s.vt_stack[:, :, idx + 1] =  F[:Vt] * s.vt_stack[:, :, idx]
end


"""
Updates stack[idx] based on stack[idx+1]
"""
function add_slice_sequence_right(s::stack, p::parameters, l::lattice, idx::Int)
  curr_Vt = copy(s.vt_stack[:, :, idx + 1])

  for slice in reverse(s.ranges[idx])
    curr_Vt = curr_Vt * slice_matrix_no_chkr(p, l, slice)
  end
  curr_Vt =  spdiagm(s.d_stack[:, idx + 1]) * curr_Vt
  F = decompose_udv!(curr_Vt)
  s.u_stack[:, :, idx] = s.u_stack[:, :, idx + 1] * F[:U]
  s.d_stack[:, idx] = F[:S]
  s.vt_stack[:, :, idx] = F[:Vt]
end


# B(slice) = exp(−1/2∆τK)exp(−∆τV(slice))exp(−1/2∆τK)
function slice_matrix_no_chkr(p::parameters, l::lattice, slice::Int, pref::Float64=1.)
  B = l.hopping_matrix_minus * interaction_matrix(p, l, slice, pref) * l.hopping_matrix_minus
end

"""
Calculates G(slice) using s.Ur,s.Dr,s.Vtr=B(M) ... B(slice) and s.Ul,s.Dl,s.Vtl=B(slice-1) ... B(1)
"""
function calculate_greens(s::stack, p::parameters, l::lattice)
  tmp = s.Vtl * s.Ur
  inner = ctranspose(s.Vtr * s.Ul) + spdiagm(s.Dl) * tmp * spdiagm(s.Dr)
  I = decompose_udv!(inner)
  s.greens = ctranspose(I[:Vt] * s.Vtr) * spdiagm(1./I[:S]) * ctranspose(s.Ul * I[:U])
end


################################################################################
# Propagation
################################################################################
function propagate(s::stack, p::parameters, l::lattice)
  if s.direction == 1
    if mod(s.current_slice, p.safe_mult) == 0
      s.current_slice += 1
      if s.current_slice == 1
        s.Ur[:], s.Dr[:], s.Vtr[:] = s.u_stack[:, :, 1], s.d_stack[:, 1], s.vt_stack[:, :, 1]
        s.u_stack[:, :, 1] = eye(Complex{Float64}, p.flv*l.sites)
        s.d_stack[:, 1] = ones(p.flv*l.sites)
        s.vt_stack[:, :, 1] = eye(Complex{Float64}, p.flv*l.sites)
        s.Ul[:], s.Dl[:], s.Vtl[:] = s.u_stack[:, :, 1], s.d_stack[:, 1], s.vt_stack[:, :, 1]
        calculate_greens(s, p, l)

      elseif s.current_slice > 1 && s.current_slice < p.slices
        idx = Int((s.current_slice - 1) / p.safe_mult)
        s.Ur[:], s.Dr[:], s.Vtr[:] = s.u_stack[:, :, idx + 1], s.d_stack[:, idx + 1], s.vt_stack[:, :, idx + 1]
        add_slice_sequence_left(s, p, l, idx)
        s.Ul[:], s.Dl[:], s.Vtl[:] = s.u_stack[:, :, idx + 1], s.d_stack[:, idx + 1], s.vt_stack[:, :, idx + 1]

        s.greens_temp[:] = s.greens[:]
        # s.greens_temp[:] = multiply_slice_matrix_right(p, l, s.greens_temp, s.current_slice - 1, -1.)
        # s.greens_temp[:] = multiply_slice_matrix_left(p, l, s.greens_temp, s.current_slice - 1, 1.)

        A = slice_matrix_no_chkr(p, l, s.current_slice -1 , -1.)
        l.temp_square = s.greens * A
        A = slice_matrix_no_chkr(p, l, s.current_slice -1, 1.)
        s.greens = A * l.temp_square

        calculate_greens(s, p, l)
        diff = maximum(diag(abs(s.greens_temp - s.greens)))
        if diff > 1e-4
          println(s.current_slice, "\t+1 Propagation stability\t", diff)
        end
      else
        idx = s.n_elements - 1
        add_slice_sequence_left(s, p, l, idx)
        s.direction = -1
        s.current_slice = p.slices + 1
        propagate(s, p, l)
      end
    else
      A = slice_matrix_no_chkr(p, l, s.current_slice, -1.)
      l.temp_square = s.greens * A
      A = slice_matrix_no_chkr(p, l, s.current_slice, 1.)
      s.greens = A * l.temp_square

      # s.greens = multiply_slice_matrix_right(p, l, s.greens, s.current_slice, -1.)
      # s.greens = multiply_slice_matrix_left(p, l, s.greens, s.current_slice, 1.)
      s.current_slice += 1
    end
  else
    if mod(s.current_slice - 1, p.safe_mult) == 0
      s.current_slice -= 1
      idx = Int(s.current_slice / p.safe_mult) + 1
      if s.current_slice == p.slices
        s.Ul[:, :], s.Dl[:], s.Vtl[:, :] = s.u_stack[:, :, end], s.d_stack[:, end], s.vt_stack[:, :, end]
        s.u_stack[:, :, end] = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
        s.d_stack[:, end] = ones(p.flv*l.sites)
        s.vt_stack[:, :, end] = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
        s.Ur[:], s.Dr[:], s.Vtr[:] = s.u_stack[:, :, end], s.d_stack[:, end], s.vt_stack[:, :, end]
        calculate_greens(s, p, l)

        # wrap gf to next slice
        A = slice_matrix_no_chkr(p, l, s.current_slice, -1.)
        l.temp_square = A * s.greens
        A = slice_matrix_no_chkr(p, l, s.current_slice, 1.)
        s.greens = l.temp_square * A
        #
        # s.greens = multiply_slice_matrix_left(p, l, s.greens, p.slices, -1.)
        # s.greens = multiply_slice_matrix_right(p, l, s.greens, p.slices, 1.)
      elseif s.current_slice > 0 && s.current_slice < p.slices
        s.greens_temp[:] = s.greens[:]
        s.Ul[:], s.Dl[:], s.Vtl[:] = s.u_stack[:, :, idx], s.d_stack[:, idx], s.vt_stack[:, :, idx]
        add_slice_sequence_right(s, p, l, idx)
        s.Ur[:], s.Dr[:], s.Vtr[:] = s.u_stack[:, :, idx], s.d_stack[:, idx], s.vt_stack[:, :, idx]
        calculate_greens(s, p, l)
        # println(real(diag(s.greens)))
        diff = maximum(diag(abs(s.greens_temp - s.greens)))
        if diff > 1e-4
          println(s.current_slice, "\t-1  Propagation stability\t", diff)
        end

        A = slice_matrix_no_chkr(p, l, s.current_slice, -1.)
        l.temp_square = A * s.greens
        A = slice_matrix_no_chkr(p, l, s.current_slice, 1.)
        s.greens = l.temp_square * A
        #
        # s.greens = multiply_slice_matrix_left(p, l, s.greens, s.current_slice, -1.)
        # s.greens = multiply_slice_matrix_right(p, l, s.greens, s.current_slice, 1.)
      elseif s.current_slice == 0
        add_slice_sequence_right(s, p, l, 1)
        s.direction = 1
        propagate(s, p, l)
      end
    else
      # wrap gf to next slice
      s.current_slice -= 1
      A = slice_matrix_no_chkr(p, l, s.current_slice, -1.)
      l.temp_square = A * s.greens
      A = slice_matrix_no_chkr(p, l, s.current_slice, 1.)
      s.greens = l.temp_square * A
      # s.greens = multiply_slice_matrix_left(p, l, s.greens, s.current_slice, -1.)
      # s.greens = multiply_slice_matrix_right(p, l, s.greens, s.current_slice, 1.)
    end
  end
  nothing
end
