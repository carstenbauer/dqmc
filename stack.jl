type stack
  u_stack::Array{Complex{Float64}, 3}
  d_stack::Array{Float64, 2}
  v_stack::Array{Complex{Float64}, 3}

  Ul::Array{Complex{Float64}, 2}
  Ur::Array{Complex{Float64}, 2}
  Dl::Array{Float64, 1}
  Dr::Array{Float64, 1}
  Vl::Array{Complex{Float64}, 2}
  Vr::Array{Complex{Float64}, 2}

  greens::Array{Complex{Float64}, 2}
  greens_temp::Array{Complex{Float64}, 2}

  U::Array{Complex{Float64}, 2}
  D::Array{Float64, 1}
  V::Array{Complex{Float64}, 2}

  # Q::Array{Complex{Float64}, 2}
  # R::Array{Complex{Float64}, 2}

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
  s.v_stack = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites, s.n_elements)

  s.greens = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.greens_temp = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)

  s.Ul = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.Ur = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.Vl = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.Vr = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.Dl = ones(Float64, p.flv*l.sites)
  s.Dr = ones(Float64, p.flv*l.sites)

  s.U = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.D = zeros(Float64, p.flv*l.sites)
  s.V = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)

  # s.Q = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  # s.R = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.ranges = UnitRange[]

  for i in 1:s.n_elements - 1
    push!(s.ranges, 1 + (i - 1) * p.safe_mult:i * p.safe_mult)
  end

end


function build_stack(s::stack, p::parameters, l::lattice)
  s.u_stack[:, :, 1] = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.d_stack[:, 1] = ones(p.flv*l.sites)
  s.v_stack[:, :, 1] = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)

  for i in 1:length(s.ranges)
    add_slice_sequence_left(s, p, l, i)
  end

  s.current_slice = p.slices + 1
  s.direction = -1
end


function add_slice_sequence_left(s::stack, p::parameters, l::lattice, idx::Int)
  curr_U = copy(s.u_stack[:, :, idx])
  # println("Adding slice seq left $idx = ", s.ranges[idx])
  for slice in s.ranges[idx]
    curr_U = slice_matrix_no_chkr(p, l, slice) * curr_U
  end

  curr_U =  curr_U * spdiagm(s.d_stack[:, idx])
  F = svdfact!(curr_U)
  s.u_stack[:, :, idx + 1] = F[:U]
  s.d_stack[:, idx + 1] = F[:S]
  s.v_stack[:, :, idx + 1] =  s.v_stack[:, :, idx] * F[:Vt]
end

# TODO
# function add_slice_sequence_right(s::stack, p::parameters, l::lattice, idx::Int)
#   curr_U = copy(s.u_stack[:, :, idx + 1])
#
#   for slice in reverse(s.ranges[idx])
#     slice_mat = slice_matrix(slice, p, l)
#     curr_U = transpose(slice_mat) * curr_U
#     # multiply_slice_matrix_left_transpose!(curr_U, l.temp_thin, slice, p, l)
#   end
#
#   s.u_stack[:, :, idx], R, p = qr(curr_U, Val{true}; thin=true)
#   # decompose_udt_alt!(slice(s.u_stack, :, :, idx), curr_U)
#
#   # for i in 1:p.particles
#   #   curr_U[:, i] *= s.d_stack[i, idx + 1]
#   # end
#
#   # s.u_stack[:, :, idx], s.d_stack[:, idx], T = decompose_udt(curr_U)
#   # s.t_stack[:, :, idx] = T * s.t_stack[:, :, idx + 1]
# end

# TODO
# function calculate_greens(s::stack, p::parameters, l::lattice)
#   A = transpose(s.Ur) * s.Ul
#   F = svdfact!(A)
#
#   s.greens = eye(l.n_sites) - s.Ul * ctranspose(F[:Vt]) * spdiagm(1. ./ F[:S]) * ctranspose(F[:U]) * transpose(s.Ur)
#   #return greens
# end


# B(slice) = exp(−1/2∆τK)exp(−∆τV(slice))exp(−1/2∆τK)
function slice_matrix_no_chkr(p::parameters, l::lattice, slice::Int, pref::Float64=1.)
  B = l.hopping_matrix_minus * interaction_matrix(p, l, slice, pref) * l.hopping_matrix_minus
end


# B(slice) = exp(−1/2∆τK)exp(−∆τV(slice))exp(−1/2∆τK)
# function slice_matrix(p::parameters, l::lattice, slice::Int, pref::Float64=1.)
#
#   M = eye(Complex{Float64}, 2*l.n_sites, 2*l.n_sites)
#
#   if pref > 0
#     M = spdiagm(interaction_matrix(p, l, slice)) * M
#     for h in l.chkr_hop
#       M = h * M
#     end
#
#     for h in reverse(l.chkr_hop)
#       M = h * M
#     end
#   else
#     for h in l.chkr_hop_inv
#       M = h * M
#     end
#
#     for h in reverse(l.chkr_hop_inv)
#       M = h * M
#     end
#     M = spdiagm(interaction_matrix(p, l, slice, -1.)) * M
#   end
#   return M
# end


################################################################################
# TODO: Propagation
################################################################################
# function propagate(s::stack, p::parameters, l::lattice)
#   if s.direction == 1
#     if mod(s.current_slice, p.safe_mult) == 0
#       s.current_slice += 1
#       if s.current_slice == 1
#         s.Ur[:], s.Dr[:], s.Tr[:] = s.u_stack[:, :, 1], s.d_stack[:, 1], s.t_stack[:, :, 1]
#         s.u_stack[:, :, 1] = l.free_fermion_wavefunction
#         s.d_stack[:, 1] = ones(size(s.d_stack)[1])
#         s.t_stack[:, :, 1] = eye(Complex{Float64}, size(s.d_stack)[1], size(s.d_stack)[1])
#         s.Ul[:], s.Dl[:], s.Tl[:] = s.u_stack[:, :, 1], s.d_stack[:, 1], s.t_stack[:, :, 1]
#         calculate_greens(s, p, l)
#
#       elseif s.current_slice > 1 && s.current_slice < p.slices
#         idx = Int((s.current_slice - 1) / p.safe_mult)
#         s.Ur[:], s.Dr[:], s.Tr[:] = s.u_stack[:, :, idx + 1], s.d_stack[:, idx + 1], s.t_stack[:, :, idx + 1]
#         add_slice_sequence_left(s, idx, p, l)
#         s.Ul[:], s.Dl[:], s.Tl[:] = s.u_stack[:, :, idx + 1], s.d_stack[:, idx + 1], s.t_stack[:, :, idx + 1]
#
#         s.greens_temp[:] = s.greens[:]
#         s.greens_temp[:] = multiply_slice_matrix_right(s.greens_temp, s.current_slice - 1, p, l, -1.)
#         s.greens_temp[:] = multiply_slice_matrix_left(s.greens_temp, s.current_slice - 1, p, l, 1.)
#
#         calculate_greens(s, p, l)
#         diff = maximum(diag(abs(s.greens_temp - s.greens)))
#         if diff > 1e-4
#           println(s.current_slice, "\t+1 Propagation stability\t", diff)
#         end
#       else
#         idx = s.n_elements - 1
#         add_slice_sequence_left(s, idx, p, l)
#         s.direction = -1
#         s.current_slice = p.slices + 1
#         propagate(s, p, l)
#       end
#     else
#       s.greens = multiply_slice_matrix_right(s.greens, s.current_slice, p, l, -1.)
#       s.greens = multiply_slice_matrix_left(s.greens, s.current_slice, p, l, 1.)
#       s.current_slice += 1
#     end
#   else
#     if mod(s.current_slice - 1, p.safe_mult) == 0
#       s.current_slice -= 1
#       idx = Int(s.current_slice / p.safe_mult) + 1
#       if s.current_slice == p.slices
#         s.Ul[:, :], s.Dl[:], s.Tl[:, :] = s.u_stack[:, :, end], s.d_stack[:, end], s.t_stack[:, :, end]
#         s.u_stack[:, :, end] = l.free_fermion_wavefunction
#         s.d_stack[:, end] = ones(p.particles)
#         s.t_stack[:, :, end] = eye(Complex{Float64}, p.particles, p.particles)
#         s.Ur[:], s.Dr[:], s.Tr[:] = s.u_stack[:, :, end], s.d_stack[:, end], s.t_stack[:, :, end]
#         calculate_greens(s, p, l)
#
#         A = slice_matrix(s.current_slice, p, l, -1.)
#         l.temp_square = A * s.greens
#         A = slice_matrix(s.current_slice, p, l, 1.)
#         s.greens = l.temp_square * A
#         #
#         # s.greens = multiply_slice_matrix_left(s.greens, p.slices, p, l, -1.)
#         # s.greens = multiply_slice_matrix_right(s.greens, p.slices, p, l, 1.)
#
#       elseif s.current_slice > 0 && s.current_slice < p.slices
#         s.greens_temp[:] = s.greens[:]
#         s.Ul[:], s.Dl[:], s.Tl[:] = s.u_stack[:, :, idx], s.d_stack[:, idx], s.t_stack[:, :, idx]
#         add_slice_sequence_right(s, idx, p, l)
#         s.Ur[:], s.Dr[:], s.Tr[:] = s.u_stack[:, :, idx], s.d_stack[:, idx], s.t_stack[:, :, idx]
#         calculate_greens(s, p, l)
#         diff = maximum(diag(abs(s.greens_temp - s.greens)))
#         if diff > 1e-4
#           println(s.current_slice, "\t-1  Propagation stability\t", diff)
#         end
#
#         A = slice_matrix(s.current_slice, p, l, -1.)
#         l.temp_square = A * s.greens
#         A = slice_matrix(s.current_slice, p, l, 1.)
#         s.greens = l.temp_square * A
#         #
#         # s.greens = multiply_slice_matrix_left(s.greens, s.current_slice, p, l, -1.)
#         # s.greens = multiply_slice_matrix_right(s.greens, s.current_slice, p, l, 1.)
#       elseif s.current_slice == 0
#         add_slice_sequence_right(s, 1, p, l)
#         s.direction = 1
#         propagate(s, p, l)
#       end
#     else
#       s.current_slice -= 1
#       A = slice_matrix(s.current_slice, p, l, -1.)
#       l.temp_square = A * s.greens
#       A = slice_matrix(s.current_slice, p, l, 1.)
#       s.greens = l.temp_square * A
#       # s.greens = multiply_slice_matrix_left(s.greens, s.current_slice, p, l, -1.)
#       # s.greens = multiply_slice_matrix_right(s.greens, s.current_slice, p, l, 1.)
#     end
#   end
# end
