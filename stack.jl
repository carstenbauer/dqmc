type stack
  u_stack::Array{Complex{Float64}, 3}
  d_stack::Array{Float64, 2}
  t_stack::Array{Complex{Float64}, 3}

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
  s.t_stack = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites, s.n_elements)

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


# function build_stack(s::stack, p::parameters, l::lattice)
#   s.u_stack[:, :, 1] = l.free_fermion_wavefunction + 0.im * l.free_fermion_wavefunction
#   s.d_stack[:, 1] = ones(size(s.d_stack)[1])
#   s.t_stack[:, :, 1] = eye(Complex{Float64}, size(s.d_stack)[1], size(s.d_stack)[1])
#
#   for i in 1:length(s.ranges)
#     add_slice_sequence_left(s, i, p, l)
#   end
#
#   s.current_slice = p.slices + 1
#   s.direction = -1
#
# end


# B(slice) = exp(−1/2∆τK)exp(−∆τV(slice))exp(−1/2∆τK)
function slice_matrix_no_chkr(p::parameters, l::lattice, slice::Int, pref::Float64=1.)
  B = l.hopping_matrix_minus * interaction_matrix(p, l, slice, pref) * l.hopping_matrix_minus
end


# B(slice) = exp(−1/2∆τK)exp(−∆τV(slice))exp(−1/2∆τK)
function slice_matrix(p::parameters, l::lattice, slice::Int, pref::Float64=1.)

  M = eye(Complex{Float64}, 2*l.n_sites, 2*l.n_sites)

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
