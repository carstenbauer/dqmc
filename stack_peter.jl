type stack
  u_stack::Array{Complex{Float64}, 3}
  d_stack::Array{Float64, 2}
  t_stack::Array{Complex{Float64}, 3}

  Ul::Array{Complex{Float64}, 2}
  Ur::Array{Complex{Float64}, 2}
  Dl::Array{Float64, 1}
  Dr::Array{Float64, 1}
  Tl::Array{Complex{Float64}, 2}
  Tr::Array{Complex{Float64}, 2}

  greens::Array{Complex{Float64}, 2}
  greens_temp::Array{Complex{Float64}, 2}

  U::Array{Complex{Float64}, 2}
  Q::Array{Complex{Float64}, 2}
  D::Array{Float64, 1}
  R::Array{Complex{Float64}, 2}
  T::Array{Complex{Float64}, 2}

  ranges::Array{UnitRange, 1}
  n_elements::Int
  current_slice::Int
  direction::Int

  stack() = new()
end


function initialize_stack(s::stack, p::parameters, l::lattice)
  s.n_elements = convert(Int, p.slices / p.safe_mult) + 1

  s.u_stack = zeros(Complex{Float64}, l.n_sites, l.n_sites, s.n_elements)
  s.d_stack = zeros(Float64, l.n_sites, s.n_elements)
  s.t_stack = zeros(Complex{Float64}, l.n_sites, l.n_sites, s.n_elements)

  s.greens = zeros(Complex{Float64}, l.n_sites, l.n_sites)
  s.greens_temp = zeros(Complex{Float64}, l.n_sites, l.n_sites)

  s.Ul = zeros(Complex{Float64}, l.n_sites, l.n_sites)
  s.Tl = eye(Complex{Float64}, l.n_sites, l.n_sites)
  s.Ur = zeros(Complex{Float64}, l.n_sites, l.n_sites)
  s.Tr = eye(Complex{Float64}, l.n_sites, l.n_sites)
  s.Dl = ones(Float64, l.n_sites)
  s.Dr = ones(Float64, l.n_sites)

  s.U = zeros(Complex{Float64}, l.n_sites, l.n_sites)
  s.Q = zeros(Complex{Float64}, l.n_sites, l.n_sites)
  s.D = zeros(Float64, l.n_sites)
  s.R = zeros(Complex{Float64}, l.n_sites, l.n_sites)
  s.T = zeros(Complex{Float64}, l.n_sites, l.n_sites)
  s.ranges = UnitRange[]

  for i in 1:s.n_elements - 1
    push!(s.ranges, 1 + (i - 1) * p.safe_mult:i * p.safe_mult)
  end
end


function build_stack(s::stack, p::parameters, l::lattice)
  s.u_stack[:, :, 1] = eye(Complex{Float64}, l.n_sites)
  s.d_stack[:, 1] = ones(l.n_sites)
  s.t_stack[:, :, 1] = eye(Complex{Float64}, l.n_sites)

  for i in 1:length(s.ranges)
    add_slice_sequence_left(s, i, p, l)
  end

  s.current_slice = p.slices + 1
  s.direction = -1

end

#
function add_slice_sequence_left(s::stack, idx::Int, p::parameters, l::lattice)
  curr_U = copy(s.u_stack[:, :, idx])

  for slice in s.ranges[idx]
    slice_mat = slice_matrix(slice, p, l)
    curr_U = slice_mat * curr_U
  end

  curr_U =  curr_U * spdiagm(s.d_stack[:, idx])
  s.u_stack[:, :, idx + 1], s.d_stack[:, idx + 1], T = decompose_udt(curr_U)
  s.t_stack[:, :, idx + 1] = T * s.t_stack[:, :, idx]
end

function add_slice_sequence_right(s::stack, idx::Int, p::parameters, l::lattice)
  curr_U = copy(s.u_stack[:, :, idx + 1])
  # println("Curr U is\n\n", curr_U)

  for slice in reverse(s.ranges[idx])
    slice_mat = slice_matrix(slice, p, l)
    curr_U = transpose(slice_mat) * curr_U
  end
  curr_U =  curr_U * spdiagm(s.d_stack[:, idx + 1])
  s.u_stack[:, :, idx], s.d_stack[:, idx], T = decompose_udt(curr_U)
  s.t_stack[:, :, idx] = T * s.t_stack[:, :, idx + 1]
end


function multiply_slice_matrix_left(M::Array{Complex{Float64}, 2}, slice::Int, p::parameters, l::lattice, pref::Float64=1.)
  A = slice_matrix(slice, p, l, pref)
  M = A * M
  return M

  if pref > 0
    M = spdiagm(interaction_matrix(p, l, slice)) * M

    for h in l.chkr_hop
      M = h * M
    end

    for h in reverse(l.chkr_hop)
      M = h * M
    end
  else
    for h in l.chkr_hop_inv
      M = h * M
    end

    for h in reverse(l.chkr_hop_inv)
      M = h * M
    end

    M = spdiagm(interaction_matrix(p, l, slice, -1.)) * M
  end
  return M
end


function multiply_slice_matrix_left!(M::Array{Complex{Float64}, 2}, temp::Array{Complex{Float64}, 2},slice::Int, p::parameters, l::lattice, pref::Float64=1.)
  mults = 0
  if pref > 0
    interaction_matrix_left!(M, p, l, slice)

    # temp[:, :] = spdiagm(interaction_matrix(p, l, slice)) * M
    # mults += 1
    for h in l.chkr_hop
      if mod(mults, 2) == 1 M[:, :] = h * temp; mults += 1
      else temp[:, :] = h * M; mults += 1 end
    end

    for h in reverse(l.chkr_hop)
      if mod(mults, 2) == 1 M[:, :] = h * temp; mults += 1
      else temp[:, :] = h * M; mults += 1 end
    end
  else
    for h in l.chkr_hop_inv
      if mod(mults, 2) == 1 M[:, :] = h * temp; mults += 1
      else temp[:, :] = h * M; mults += 1 end
    end

    for h in reverse(l.chkr_hop_inv)
      if mod(mults, 2) == 1 M[:, :] = h * temp; mults += 1
      else temp[:, :] = h * M; mults += 1 end
    end

    if mod(mults, 2) == 1 interaction_matrix_left!(temp, p, l, slice, -1.)
    else interaction_matrix_left!(M, p, l, slice, -1.) end
  end
  if mod(mults, 2) == 1
    M[:, :] = temp[:, :]
  end
end

function multiply_slice_matrix_left_transpose!(M::Array{Complex{Float64}, 2}, temp::Array{Complex{Float64}, 2},slice::Int, p::parameters, l::lattice, pref::Float64=1.)
  mults = 0

  if pref > 0
    for h in l.chkr_hop
      if mod(mults, 2) == 1 M[:, :] = h * temp; mults += 1
      else temp[:, :] = h * M; mults += 1 end
    end

    for h in reverse(l.chkr_hop)
      if mod(mults, 2) == 1 M[:, :] = h * temp; mults += 1
      else temp[:, :] = h * M; mults += 1 end
    end

    if mod(mults, 2) == 1 interaction_matrix_left!(temp, p, l, slice)
    else interaction_matrix_left!(M, p, l, slice) end

  else
    if mod(mults, 2) == 1 interaction_matrix_left!(temp, p, l, slice, -1.)
    else interaction_matrix_left!(M, p, l, slice, -1.) end

    for h in l.chkr_hop_inv
      if mod(mults, 2) == 1 M[:, :] = h * temp; mults += 1
      else temp[:, :] = h * M; mults += 1 end
    end

    for h in reverse(l.chkr_hop_inv)
      if mod(mults, 2) == 1 M[:, :] = h * temp; mults += 1
      else temp[:, :] = h * M; mults += 1 end
    end
  end

  if mod(mults, 2) == 1
    M[:, :] = temp[:, :]
  end
end


function multiply_slice_matrix_right!(M::Array{Complex{Float64}, 2}, temp::Array{Complex{Float64}}, slice::Int,
  p::parameters, l::lattice, pref::Float64=1.)
  mults = 0

  if pref > 0
    for h in l.chkr_hop
      if mults == 1 M[:, :] = temp * h; mults = 0
      else
        temp[:, :] = M * h; mults = 1 end
    end

    for h in reverse(l.chkr_hop)
      if mults == 1 M[:, :] = temp * h; mults = 0
      else temp[:, :] = M * h; mults = 1 end
    end

    if mults == 1 interaction_matrix_right!(temp, p, l, slice)
    else interaction_matrix_right!(M, p, l, slice) end
  else
    if mults == 1 interaction_matrix_right!(temp, p, l, slice, -1.)
    else interaction_matrix_right!(M, p, l, slice, -1.) end

    # if mults == 1 M[:, :] = temp * spdiagm(interaction_matrix(p, l, slice, -1.)); mults = 0
    # else temp[:, :] = M * spdiagm(interaction_matrix(p, l, slice, -1.)); mults = 1 end

    for h in l.chkr_hop_inv
      if mults == 1 M[:, :] = temp * h; mults = 0
      else temp[:, :] = M * h; mults = 1 end
    end

    for h in reverse(l.chkr_hop_inv)
      if mults == 1 M[:, :] = temp * h; mults = 0
      else temp[:, :] = M * h; mults = 1 end
    end
  end

  if mults == 1 M[:, :] = temp[:, :] end
end


function multiply_slice_matrix_right(M::Array{Complex{Float64}, 2}, slice::Int,
  p::parameters, l::lattice, pref::Float64=1.)
  A = slice_matrix(slice, p, l, pref)
  M = M * A
  return M

  if pref > 0
    for h in l.chkr_hop
      M = M * h
    end

    for h in reverse(l.chkr_hop)
      M = M * h
    end

    M = M * spdiagm(interaction_matrix(p, l, slice))
  else
    M = M * spdiagm(interaction_matrix(p, l, slice, -1.))

    for h in l.chkr_hop_inv
      M = M * h
    end

    for h in reverse(l.chkr_hop_inv)
      M = M * h
    end
  end
  return M
end

function slice_matrix(slice::Int, p::parameters, l::lattice, pref::Float64=1.)

  M = eye(Complex{Float64}, l.n_sites, l.n_sites)

  if pref > 0
    M = spdiagm(interaction_matrix(p, l, slice)) * M
    for h in l.chkr_hop
      M = h * M
    end

    for h in reverse(l.chkr_hop)
      M = h * M
    end
  else
    for h in l.chkr_hop_inv
      M = h * M
    end

    for h in reverse(l.chkr_hop_inv)
      M = h * M
    end
    M = spdiagm(interaction_matrix(p, l, slice, -1.)) * M
  end
  return M
end

function calculate_greens(s::stack, p::parameters, l::lattice)
    # println(s.Tl)
  A = spdiagm(s.Dl) * (s.Tl * transpose(s.Tr)) * spdiagm(s.Dr)

  M, S, N = decompose_udt(A)

  U = s.Ul * M
  D = S
  T = N * transpose(s.Ur)

  # A' * X = B
  # X = A'-1 * B
  # X' = B' * A^-1

  inside = ctranspose(\(ctranspose(T), U)) + diagm(D)
  Ui, Di, Ti = decompose_udt(inside)
  Di_inv = 1. ./ Di

  U_left = U * Ui
  T_left = spdiagm(Di) * Ti * T
  s.greens = \(T_left, ctranspose(U_left))
  # inside = ctranspose(U) * ctranspose(T) + diagm(D)
  # Fi = svdfact!(inside)
  # Ui = Fi[:U]
  # Di_inv = 1./Fi[:S]
  # Ti = Fi[:Vt]

  # s.greens = ctranspose(T) * ctranspose(Ti) * spdiagm(Di_inv) * ctranspose(Ui) * ctranspose(U)

end

################################################################################
# Propagation
################################################################################
function propagate(s::stack, p::parameters, l::lattice)
  if s.direction == 1
    if mod(s.current_slice, p.safe_mult) == 0
      s.current_slice += 1
      if s.current_slice == 1
        s.Ur[:], s.Dr[:], s.Tr[:] = s.u_stack[:, :, 1], s.d_stack[:, 1], s.t_stack[:, :, 1]
        s.u_stack[:, :, 1] = eye(Complex{Float64}, l.n_sites)
        s.d_stack[:, 1] = ones(l.n_sites)
        s.t_stack[:, :, 1] = eye(Complex{Float64}, l.n_sites)
        s.Ul[:], s.Dl[:], s.Tl[:] = s.u_stack[:, :, 1], s.d_stack[:, 1], s.t_stack[:, :, 1]
        calculate_greens(s, p, l)

      elseif s.current_slice > 1 && s.current_slice < p.slices
        idx = Int((s.current_slice - 1) / p.safe_mult)
        s.Ur[:], s.Dr[:], s.Tr[:] = s.u_stack[:, :, idx + 1], s.d_stack[:, idx + 1], s.t_stack[:, :, idx + 1]
        add_slice_sequence_left(s, idx, p, l)
        s.Ul[:], s.Dl[:], s.Tl[:] = s.u_stack[:, :, idx + 1], s.d_stack[:, idx + 1], s.t_stack[:, :, idx + 1]

        s.greens_temp[:] = s.greens[:]
        s.greens_temp[:] = multiply_slice_matrix_right(s.greens_temp, s.current_slice - 1, p, l, -1.)
        s.greens_temp[:] = multiply_slice_matrix_left(s.greens_temp, s.current_slice - 1, p, l, 1.)

        calculate_greens(s, p, l)
        diff = maximum(diag(abs(s.greens_temp - s.greens)))
        if diff > 1e-4
          println(s.current_slice, "\t+1 Propagation stability\t", diff)
        end
      else
        idx = s.n_elements - 1
        add_slice_sequence_left(s, idx, p, l)
        s.direction = -1
        s.current_slice = p.slices + 1
        propagate(s, p, l)
      end
    else
      s.greens = multiply_slice_matrix_right(s.greens, s.current_slice, p, l, -1.)
      s.greens = multiply_slice_matrix_left(s.greens, s.current_slice, p, l, 1.)
      s.current_slice += 1
    end
  else
    if mod(s.current_slice - 1, p.safe_mult) == 0
      s.current_slice -= 1
      idx = Int(s.current_slice / p.safe_mult) + 1
      if s.current_slice == p.slices
        s.Ul[:, :], s.Dl[:], s.Tl[:, :] = s.u_stack[:, :, end], s.d_stack[:, end], s.t_stack[:, :, end]
        s.u_stack[:, :, end] = eye(Complex{Float64}, l.n_sites, l.n_sites)
        s.d_stack[:, end] = ones(l.n_sites)
        s.t_stack[:, :, end] = eye(Complex{Float64}, l.n_sites, l.n_sites)
        s.Ur[:], s.Dr[:], s.Tr[:] = s.u_stack[:, :, end], s.d_stack[:, end], s.t_stack[:, :, end]
        calculate_greens(s, p, l)

        A = slice_matrix(s.current_slice, p, l, -1.)
        l.temp_square = A * s.greens
        A = slice_matrix(s.current_slice, p, l, 1.)
        s.greens = l.temp_square * A
        #
        # s.greens = multiply_slice_matrix_left(s.greens, p.slices, p, l, -1.)
        # s.greens = multiply_slice_matrix_right(s.greens, p.slices, p, l, 1.)

      elseif s.current_slice > 0 && s.current_slice < p.slices
        s.greens_temp[:] = s.greens[:]
        s.Ul[:], s.Dl[:], s.Tl[:] = s.u_stack[:, :, idx], s.d_stack[:, idx], s.t_stack[:, :, idx]
        add_slice_sequence_right(s, idx, p, l)
        s.Ur[:], s.Dr[:], s.Tr[:] = s.u_stack[:, :, idx], s.d_stack[:, idx], s.t_stack[:, :, idx]
        calculate_greens(s, p, l)
        # println(real(diag(s.greens)))
        diff = maximum(diag(abs(s.greens_temp - s.greens)))
        if diff > 1e-4
          println(s.current_slice, "\t-1  Propagation stability\t", diff)
        end

        A = slice_matrix(s.current_slice, p, l, -1.)
        l.temp_square = A * s.greens
        A = slice_matrix(s.current_slice, p, l, 1.)
        s.greens = l.temp_square * A
        #
        # s.greens = multiply_slice_matrix_left(s.greens, s.current_slice, p, l, -1.)
        # s.greens = multiply_slice_matrix_right(s.greens, s.current_slice, p, l, 1.)
      elseif s.current_slice == 0
        add_slice_sequence_right(s, 1, p, l)
        s.direction = 1
        propagate(s, p, l)
      end
    else
      s.current_slice -= 1
      A = slice_matrix(s.current_slice, p, l, -1.)
      l.temp_square = A * s.greens
      A = slice_matrix(s.current_slice, p, l, 1.)
      s.greens = l.temp_square * A
      # s.greens = multiply_slice_matrix_left(s.greens, s.current_slice, p, l, -1.)
      # s.greens = multiply_slice_matrix_right(s.greens, s.current_slice, p, l, 1.)
    end
  end
end
