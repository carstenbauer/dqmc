mutable struct Stack
  u_stack::Array{Complex{Float64}, 3}
  d_stack::Array{Float64, 2}
  t_stack::Array{Complex{Float64}, 3}

  Ul::Array{Complex{Float64}, 2}
  Ur::Array{Complex{Float64}, 2}
  Dl::Array{Float64, 1}
  Dr::Array{Float64, 1}
  Tl::Array{Complex{Float64}, 2}
  Tr::Array{Complex{Float64}, 2}

  Ui::Array{Complex{Float64}, 2}
  Di::Array{Float64, 1}
  Ti::Array{Complex{Float64}, 2}
  U::Array{Complex{Float64}, 2}
  D::Array{Float64, 1}
  T::Array{Complex{Float64}, 2}

  greens::Array{Complex{Float64}, 2}
  greens_temp::Array{Complex{Float64}, 2}

  delta_i::Array{Complex{Float64}, 2}
  M::Array{Complex{Float64}, 2}

  eye_flv::Array{Complex{Float64},2}
  eye_full::Array{Complex{Float64},2}

  ranges::Array{UnitRange, 1}
  n_elements::Int
  current_slice::Int
  direction::Int

  Stack() = new()
end


function initialize_stack(s::Stack, p::Parameters, l::Lattice)
  s.n_elements = convert(Int, p.slices / p.safe_mult) + 1

  s.u_stack = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites, s.n_elements)
  s.d_stack = zeros(Float64, p.flv*l.sites, s.n_elements)
  s.t_stack = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites, s.n_elements)

  s.greens = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  # s.greens_svs = zeros(p.flv*l.sites)
  s.greens_temp = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)

  s.Ul = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.Ur = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.Tl = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.Tr = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.Dl = ones(Float64, p.flv*l.sites)
  s.Dr = ones(Float64, p.flv*l.sites)


  s.Ui = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.Di = zeros(Float64, p.flv*l.sites)
  s.Ti = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.U = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.D = zeros(Float64, p.flv*l.sites)
  s.T = zeros(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)


  s.delta_i = zeros(Complex{Float64}, p.flv, p.flv)
  s.M = zeros(Complex{Float64}, p.flv, p.flv)

  s.eye_flv = eye(p.flv,p.flv)
  s.eye_full = eye(p.flv*l.sites,p.flv*l.sites)

  s.ranges = UnitRange[]

  for i in 1:s.n_elements - 1
    push!(s.ranges, 1 + (i - 1) * p.safe_mult:i * p.safe_mult)
  end

end


function build_stack(s::Stack, p::Parameters, l::Lattice)
  println("Building stack")
  s.u_stack[:, :, 1] = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  s.d_stack[:, 1] = ones(p.flv*l.sites)
  s.t_stack[:, :, 1] = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)

  @inbounds for i in 1:length(s.ranges)
    add_slice_sequence_left(s, p, l, i)
  end

  s.current_slice = p.slices + 1
  s.direction = -1
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
  s.u_stack[:, :, idx + 1], s.d_stack[:, idx + 1], T = decompose_udt(curr_U)
  s.t_stack[:, :, idx + 1] =  T * s.t_stack[:, :, idx]
end


"""
Updates stack[idx] based on stack[idx+1]
"""
function add_slice_sequence_right(s::Stack, p::Parameters, l::Lattice, idx::Int)
  curr_U = copy(s.u_stack[:, :, idx + 1])

  for slice in reverse(s.ranges[idx])
    curr_U = ctranspose(slice_matrix(p, l, slice)) * curr_U
  end
  curr_U =  curr_U * spdiagm(s.d_stack[:, idx + 1])
  s.u_stack[:, :, idx], s.d_stack[:, idx], T = decompose_udt(curr_U)
  s.t_stack[:, :, idx] = T * s.t_stack[:, :, idx + 1]
end


# B(slice) = exp(−1/2∆τK)exp(−∆τV(slice))exp(−1/2∆τK)
function slice_matrix_no_chkr(p::Parameters, l::Lattice, slice::Int, power::Float64=1.)
  if power > 0
    return l.hopping_matrix_exp * interaction_matrix_exp(p, l, slice, power) * l.hopping_matrix_exp
  else
    return l.hopping_matrix_exp_inv * interaction_matrix_exp(p, l, slice, power) * l.hopping_matrix_exp_inv
  end
end


"""
Calculates G(slice) using s.Ur,s.Dr,s.Tr=B(M) ... B(slice) and s.Ul,s.Dl,s.Tl=B(slice-1) ... B(1)
"""
function calculate_greens(s::Stack, p::Parameters, l::Lattice)
  inner = spdiagm(s.Dl) * (s.Tl * ctranspose(s.Tr)) * spdiagm(s.Dr)
  s.Ui, s.Di, s.Ti = decompose_udt(inner)

  s.U = s.Ul * s.Ui
  s.T = s.Ti * ctranspose(s.Ur)
  s.Ui, s.Di, s.Ti = decompose_udt(/(ctranspose(s.U),s.T) + spdiagm(s.Di))

  s.greens =  \(spdiagm(s.Di) * (s.Ti * s.T) , ctranspose(s.U * s.Ui))
end

function calculate_greens_svd(s::Stack, p::Parameters, l::Lattice)
  tmp = s.Tl * ctranspose(s.Tr)
  inner = ctranspose(s.Ul) * s.Ur + spdiagm(s.Dl) * tmp * spdiagm(s.Dr)
  s.U, s.D, s.T = decompose_udv(inner)
  # s.greens_svs = 1./s.D
  s.greens = (s.Ur * ctranspose(s.T)) * spdiagm(1./s.D) * ctranspose(s.Ul * s.U)
end


################################################################################
# Propagation
################################################################################
function propagate(s::Stack, p::Parameters, l::Lattice)
  if s.direction == 1
    if mod(s.current_slice, p.safe_mult) == 0
      s.current_slice += 1
      if s.current_slice == 1
        s.Ur[:], s.Dr[:], s.Tr[:] = s.u_stack[:, :, 1], s.d_stack[:, 1], s.t_stack[:, :, 1]
        s.u_stack[:, :, 1] = eye(Complex{Float64}, p.flv*l.sites)
        s.d_stack[:, 1] = ones(p.flv*l.sites)
        s.t_stack[:, :, 1] = eye(Complex{Float64}, p.flv*l.sites)
        s.Ul[:], s.Dl[:], s.Tl[:] = s.u_stack[:, :, 1], s.d_stack[:, 1], s.t_stack[:, :, 1]
        @time calculate_greens(s, p, l);
        @time calculate_greens_svd(s, p, l);

      elseif s.current_slice > 1 && s.current_slice < p.slices
        idx = Int((s.current_slice - 1) / p.safe_mult)
        s.Ur[:], s.Dr[:], s.Tr[:] = s.u_stack[:, :, idx + 1], s.d_stack[:, idx + 1], s.t_stack[:, :, idx + 1]
        add_slice_sequence_left(s, p, l, idx)
        s.Ul[:], s.Dl[:], s.Tl[:] = s.u_stack[:, :, idx + 1], s.d_stack[:, idx + 1], s.t_stack[:, :, idx + 1]

        s.greens_temp = copy(s.greens)
        multiply_slice_matrix_inv_right!(p, l, s.current_slice - 1, s.greens_temp)
        multiply_slice_matrix_left!(p, l, s.current_slice - 1, s.greens_temp)

        @time calculate_greens(s, p, l);
        @time calculate_greens_svd(s, p, l);
        diff = maximum(abs.(s.greens_temp - s.greens))
        # if diff > 1e-4
          # @printf("%d \t+1 Propagation stability\t %.4f\n", s.current_slice, diff)
        # end
      else
        idx = s.n_elements - 1
        add_slice_sequence_left(s, p, l, idx)
        s.direction = -1
        s.current_slice = p.slices + 1
        propagate(s, p, l)
      end
    else
      multiply_slice_matrix_inv_right!(p, l, s.current_slice, s.greens)
      multiply_slice_matrix_left!(p, l, s.current_slice, s.greens)
      s.current_slice += 1
    end
  else
    if mod(s.current_slice - 1, p.safe_mult) == 0
      s.current_slice -= 1
      idx = Int(s.current_slice / p.safe_mult) + 1
      if s.current_slice == p.slices
        s.Ul[:, :], s.Dl[:], s.Tl[:, :] = s.u_stack[:, :, end], s.d_stack[:, end], s.t_stack[:, :, end]
        s.u_stack[:, :, end] = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
        s.d_stack[:, end] = ones(p.flv*l.sites)
        s.t_stack[:, :, end] = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
        s.Ur[:], s.Dr[:], s.Tr[:] = s.u_stack[:, :, end], s.d_stack[:, end], s.t_stack[:, :, end]
        @time calculate_greens(s, p, l);
        @time calculate_greens_svd(s, p, l);

        # wrap gf to next slice
        multiply_slice_matrix_inv_left!(p, l, p.slices, s.greens)
        multiply_slice_matrix_right!(p, l, p.slices, s.greens)

      elseif s.current_slice > 0 && s.current_slice < p.slices
        s.greens_temp[:] = s.greens[:]
        s.Ul[:], s.Dl[:], s.Tl[:] = s.u_stack[:, :, idx], s.d_stack[:, idx], s.t_stack[:, :, idx]
        add_slice_sequence_right(s, p, l, idx)
        s.Ur[:], s.Dr[:], s.Tr[:] = s.u_stack[:, :, idx], s.d_stack[:, idx], s.t_stack[:, :, idx]
        @time calculate_greens(s, p, l);
        @time calculate_greens_svd(s, p, l);
        # println(real(diag(s.greens)))
        diff = maximum(abs.(s.greens_temp - s.greens))
        # if diff > 1e-4
          # @printf("%d \t-1 Propagation stability\t %.4f\n", s.current_slice, diff)
        # end

        multiply_slice_matrix_inv_left!(p, l, s.current_slice, s.greens)
        multiply_slice_matrix_right!(p, l, s.current_slice, s.greens)

      elseif s.current_slice == 0
        add_slice_sequence_right(s, p, l, 1)
        s.direction = 1
        propagate(s, p, l)
      end
    else
      # wrap gf to next slice
      s.current_slice -= 1
      multiply_slice_matrix_inv_left!(p, l, s.current_slice, s.greens)
      multiply_slice_matrix_right!(p, l, s.current_slice, s.greens)
    end
  end
  nothing
end
