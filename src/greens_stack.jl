mutable struct GreensStack{G<:Number} # G = GreensEltype
  n_elements::Int
  ranges::Array{UnitRange, 1}
  current_slice::Int # running internally over 0:p.slices+1, where 0 and p.slices+1 are artifcial to prepare next sweep direction.
  direction::Int

  eye_flv::Matrix{Float64} # UPGRADE: Replace by dynamically sized I everywhere
  eye_full::Matrix{Float64} # UPGRADE: Replace by dynamically sized I everywhere
  ones_vec::Vector{Float64}

  greens::Matrix{G}
  log_det::Float64 # contains logdet of greens_{p.slices+1} === greens_1
                            # after we calculated a fresh greens in propagate()

  C::Vector{G}
  S::Vector{G}
  R::Vector{G}
  eV::SparseMatrixCSC{G,Int64}

  Bl::Matrix{G}

  # unsure about those -> always allocate them
  curr_U::Matrix{G}
  D::Vector{Float64}
  U::Matrix{G}
  T::Matrix{G}
  d::Vector{Float64}
  tmp::Matrix{G}
  tmp2::Matrix{G}

  # global update
  gb_u_stack::Array{G, 3}
  gb_d_stack::Matrix{Float64}
  gb_t_stack::Array{G, 3}

  gb_greens::Matrix{G}
  gb_log_det::Float64

  gb_hsfield::Array{Float64, 3}

  # etgreens stack
  u_stack::Array{G, 3}
  d_stack::Matrix{Float64}
  t_stack::Array{G, 3}

  # calc detratio
  delta_i::Matrix{G}
  M::Matrix{G}
  eVop1::Matrix{G}
  eVop2::Matrix{G}
  eVop1eVop2::Matrix{G}
  Mtmp::Matrix{G}
  Mtmp2::Matrix{G}

  # calc greens
  Ul::Matrix{G}
  Ur::Matrix{G}
  Dl::Vector{Float64}
  Dr::Vector{Float64}
  Tl::Matrix{G}
  Tr::Matrix{G}

  # update greens
  A::Matrix{G}
  B::Matrix{G}
  AB::Matrix{G}

  # propagate
  greens_temp::Matrix{G}


  GreensStack{G}() where G = new{G}()
end






@def stackshortcuts begin
  g = mc.g
  N = mc.l.sites
  flv = mc.p.flv
  safe_mult = mc.p.safe_mult
  G = geltype(mc)
end


function allocate_global_update!(mc)
  @stackshortcuts
  # Global update backup
  g.gb_u_stack = zero(g.u_stack)
  g.gb_d_stack = zero(g.d_stack)
  g.gb_t_stack = zero(g.t_stack)
  g.gb_greens = zero(g.greens)
  g.gb_log_det = 0.
  g.gb_hsfield = zero(mc.p.hsfield)
end

function allocate_etgreens_stack!(mc)
  @stackshortcuts
  g.u_stack = zeros(G, flv*N, flv*N, g.n_elements)
  g.d_stack = zeros(Float64, flv*N, g.n_elements)
  g.t_stack = zeros(G, flv*N, flv*N, g.n_elements)
end

function allocate_calc_detratio!(mc)
  @stackshortcuts
  ## calc_detratio
  g.M = zeros(G, flv, flv)
  g.Mtmp = g.eye_flv - g.greens[1:N:end,1:N:end]
  g.delta_i = zeros(G, size(g.eye_flv))
  g.Mtmp2 = zeros(G, size(g.eye_flv))
  g.eVop1eVop2 = zeros(G, size(g.eye_flv))
  g.eVop1 = zeros(G, flv, flv)
  g.eVop2 = zeros(G, flv, flv)
end

function allocate_calc_greens!(mc)
  @stackshortcuts
  g.Ul = Matrix{G}(I, flv*N, flv*N)
  g.Ur = Matrix{G}(I, flv*N, flv*N)
  g.Tl = Matrix{G}(I, flv*N, flv*N)
  g.Tr = Matrix{G}(I, flv*N, flv*N)
  g.Dl = ones(Float64, flv*N)
  g.Dr = ones(Float64, flv*N)
end

function allocate_update_greens!(mc)
  @stackshortcuts
  ## update_greens
  g.A = g.greens[:,1:N:end]
  g.B = g.greens[1:N:end,:]
  g.AB = g.A * g.B
end

function allocate_propagate!(mc)
  @stackshortcuts
  g.greens_temp = zeros(G, flv*N, flv*N)
end

function _initialize_stack(mc::AbstractDQMC)
  @stackshortcuts
  g.n_elements = convert(Int, mc.p.slices / safe_mult) + 1

  g.ranges = UnitRange[]
  for i in 1:g.n_elements - 1
    push!(g.ranges, 1 + (i - 1) * safe_mult:i * safe_mult)
  end

  g.eye_flv = Matrix{Float64}(I, flv, flv)
  g.eye_full = Matrix{Float64}(I, flv*N,flv*N)
  g.ones_vec = ones(flv*N)

  g.greens = zeros(G, flv*N, flv*N)

  # interaction matrix
  g.C = zeros(G, N)
  g.S = zeros(G, N)
  g.R = zeros(G, N)
  g.eV = spzeros(G, flv*N, flv*N)

  # slice matrix
  cbtype(mc) === CBFalse && (g.Bl = zeros(G, flv*N, flv*N))

  # unsure about those
  g.curr_U = zeros(G, flv*N, flv*N) # used in tdgf, add_slice_sequence_ and propagate
  g.D = zeros(Float64, flv*N) # used in calc_greens in mc (below) and in calc greens in fermion meas.
  g.d = zeros(Float64, flv*N) # same as above
  g.tmp = zeros(G, flv*N, flv*N)
  g.tmp2 = zeros(G, flv*N, flv*N)
  g.U = zeros(G, flv*N, flv*N)
  g.T = zeros(G, flv*N, flv*N)
end

function initialize_stack(mc::AbstractDQMC)
  @mytimeit mc.a.to "initialize_stack" begin
    _initialize_stack(mc)

    # allocate for dqmc
    allocate_etgreens_stack!(mc)
    allocate_global_update!(mc)
    allocate_calc_detratio!(mc)
    allocate_calc_greens!(mc)
    allocate_update_greens!(mc)
    allocate_propagate!(mc)

    # allocate for measurements during dqmc
    # allocate_etpc!(mc)

  end #timeit

  nothing
end








function build_stack(mc::AbstractDQMC)
  p = mc.p
  g = mc.g
  a = mc.a

  @mytimeit a.to "build_stack" begin

  g.u_stack[:, :, 1] = g.eye_full
  g.d_stack[:, 1] = g.ones_vec
  g.t_stack[:, :, 1] = g.eye_full

  @inbounds for i in 1:length(g.ranges)
    add_slice_sequence_left(mc, i)
  end

  g.current_slice = p.slices + 1
  g.direction = -1

  end #timeit

  nothing
end


"""
Updates stack[idx+1] based on stack[idx]
"""
function add_slice_sequence_left(mc::AbstractDQMC, idx::Int)
  g = mc.g
  a = mc.a

  copyto!(g.curr_U, g.u_stack[:, :, idx])

  # println("Adding slice seq left $idx = ", g.ranges[idx])
  for slice in g.ranges[idx]
    multiply_B_left!(mc, slice, g.curr_U)
  end

  @views rmul!(g.curr_U, Diagonal(g.d_stack[:, idx]))
  @mytimeit a.to "decompose_udt!" g.u_stack[:, :, idx + 1], T = @views decompose_udt!(g.curr_U, g.d_stack[:, idx + 1])
  @views mul!(g.t_stack[:, :, idx + 1],  T, g.t_stack[:, :, idx])
  nothing
end


"""
Updates stack[idx] based on stack[idx+1]
"""
function add_slice_sequence_right(mc::AbstractDQMC, idx::Int)
  g = mc.g
  a = mc.a

  copyto!(g.curr_U, g.u_stack[:, :, idx + 1])

  for slice in reverse(g.ranges[idx])
    multiply_daggered_B_left!(mc, slice, g.curr_U)
  end

  @views rmul!(g.curr_U, Diagonal(g.d_stack[:, idx + 1]))
  @mytimeit a.to "decompose_udt!" g.u_stack[:, :, idx], T = @views decompose_udt!(g.curr_U, g.d_stack[:, idx])
  @views mul!(g.t_stack[:, :, idx], T, g.t_stack[:, :, idx + 1])
  nothing
end


function wrap_greens!(mc::AbstractDQMC, gf::Matrix, curr_slice::Int=mc.g.current_slice, direction::Int=mc.g.direction)
  if direction == -1
    multiply_B_inv_left!(mc, curr_slice - 1, gf)
    multiply_B_right!(mc, curr_slice - 1, gf)
  else
    multiply_B_left!(mc, curr_slice, gf)
    multiply_B_inv_right!(mc, curr_slice, gf)
  end
  nothing
end


function wrap_greens(mc::AbstractDQMC, gf::Matrix,slice::Int=mc.g.current_slice,direction::Int=mc.g.direction)
  temp = copy(gf)
  wrap_greens!(mc, temp, slice, direction)
  return temp
end


"""
Calculates G(slice) using g.Ur,g.Dr,g.Tr=B(slice)' ... B(M)' and g.Ul,g.Dl,g.Tl=B(slice-1) ... B(1)
"""
function calculate_greens(mc::AbstractDQMC)
  # TODO: Use inv_one_plus_two_udts from linalg.jl here
  g = mc.g
  tmp = mc.g.tmp
  tmp2 = mc.g.tmp2
  a = mc.a
  @mytimeit a.to "calculate_greens" begin

  mul!(tmp, g.Tl, adjoint(g.Tr))
  rmul!(tmp, Diagonal(g.Dr))
  lmul!(Diagonal(g.Dl), tmp)
  @mytimeit a.to "decompose_udt!" g.U, g.T = decompose_udt!(tmp, g.D)

  mul!(tmp, g.Ul, g.U)
  g.U .= tmp
  mul!(tmp2, g.T, adjoint(g.Ur))
  myrdiv!(tmp, g.U', tmp2) # or tmp = g.U' / tmp2 or mul!(tmp, adjoint(g.U), inv(tmp2))
  tmp[diagind(tmp)] .+= g.D
  @mytimeit a.to "decompose_udt!" u, t = decompose_udt!(tmp, g.d)

  mul!(tmp, t, tmp2)
  g.T = inv(tmp)
  mul!(tmp, g.U, u)
  g.U = adjoint(tmp)
  g.d .= 1 ./ g.d

  copyto!(tmp2, g.U)
  lmul!(Diagonal(g.d), tmp2)
  mul!(g.greens, g.T, tmp2)
  end #timeit
  nothing
end




"""
Only reasonable immediately after calculate_greens() because it depends on g.U, g.d and g.T!
"""
function calculate_logdet(mc::AbstractDQMC)
  g = mc.g

  if mc.p.opdim == 1
    g.log_det = real(log(complex(det(g.U))) + sum(log.(g.d)) + log(complex(det(g.T))))
  else
    g.log_det = real(logdet(g.U) + sum(log.(g.d)) + logdet(g.T))
  end
end


################################################################################
# Propagation
################################################################################
function propagate(mc::AbstractDQMC)
  g = mc.g
  p = mc.p

  if g.direction == 1
    if mod(g.current_slice, p.safe_mult) == 0
      g.current_slice +=1 # slice we are going to
      # println("we are going to $(g.current_slice) with a fresh gf.")
      if g.current_slice == 1
        g.Ur[:, :], g.Dr[:], g.Tr[:, :] = g.u_stack[:, :, 1], g.d_stack[:, 1], g.t_stack[:, :, 1]
        g.u_stack[:, :, 1] = g.eye_full
        g.d_stack[:, 1] = g.ones_vec
        g.t_stack[:, :, 1] = g.eye_full
        g.Ul[:,:], g.Dl[:], g.Tl[:,:] = g.u_stack[:, :, 1], g.d_stack[:, 1], g.t_stack[:, :, 1]

        calculate_greens(mc) # greens_1 ( === greens_{m+1} )
        calculate_logdet(mc)

      elseif 1 < g.current_slice <= p.slices
        idx = Int((g.current_slice - 1)/p.safe_mult)

        g.Ur[:, :], g.Dr[:], g.Tr[:, :] = g.u_stack[:, :, idx+1], g.d_stack[:, idx+1], g.t_stack[:, :, idx+1]
        add_slice_sequence_left(mc, idx)
        g.Ul[:,:], g.Dl[:], g.Tl[:,:] = g.u_stack[:, :, idx+1], g.d_stack[:, idx+1], g.t_stack[:, :, idx+1]

        if p.all_checks
          copyto!(g.greens_temp, g.greens)
        end

        wrap_greens!(mc, g.greens_temp, g.current_slice - 1, 1)

        calculate_greens(mc) # greens_{slice we are propagating to}

        if p.all_checks
          diff = maximum(absdiff(g.greens_temp, g.greens))
          if diff > 1e-7
            @printf("->%d \t+1 Propagation instability\t %.4f\n", g.current_slice, diff)
          end
        end

      else # we are going to p.slices+1
        idx = g.n_elements - 1
        add_slice_sequence_left(mc, idx)
        g.direction = -1
        g.current_slice = p.slices+1 # redundant
        propagate(mc)
      end

    else
      # Wrapping
      wrap_greens!(mc, g.greens, g.current_slice, 1)

      g.current_slice += 1
    end

  else # g.direction == -1
    if mod(g.current_slice-1, p.safe_mult) == 0
      g.current_slice -= 1 # slice we are going to
      # println("we are going to $(g.current_slice) with a fresh+wrapped gf.")
      if g.current_slice == p.slices
        g.Ul[:, :], g.Dl[:], g.Tl[:, :] = g.u_stack[:, :, end], g.d_stack[:, end], g.t_stack[:, :, end]
        g.u_stack[:, :, end] = g.eye_full
        g.d_stack[:, end] = g.ones_vec
        g.t_stack[:, :, end] = g.eye_full
        g.Ur[:,:], g.Dr[:], g.Tr[:,:] = g.u_stack[:, :, end], g.d_stack[:, end], g.t_stack[:, :, end]

        calculate_greens(mc) # greens_{p.slices+1} === greens_1
        calculate_logdet(mc) # calculate logdet for potential global update

        # wrap to greens_{p.slices}
        wrap_greens!(mc, g.greens, g.current_slice + 1, -1)

      elseif 0 < g.current_slice < p.slices
        idx = Int(g.current_slice / p.safe_mult) + 1
        g.Ul[:, :], g.Dl[:], g.Tl[:, :] = g.u_stack[:, :, idx], g.d_stack[:, idx], g.t_stack[:, :, idx]
        add_slice_sequence_right(mc, idx)
        g.Ur[:,:], g.Dr[:], g.Tr[:,:] = g.u_stack[:, :, idx], g.d_stack[:, idx], g.t_stack[:, :, idx]

        if p.all_checks
          copyto!(g.greens_temp, g.greens)
        end

        calculate_greens(mc)

        if p.all_checks
          diff = maximum(absdiff(g.greens_temp, g.greens))
          if diff > 1e-7
            @printf("->%d \t-1 Propagation instability\t %.4f\n", g.current_slice, diff)
          end
        end

        wrap_greens!(mc, g.greens, g.current_slice + 1, -1)

      else # we are going to 0
        idx = 1
        add_slice_sequence_right(mc, idx)
        g.direction = 1
        g.current_slice = 0 # redundant
        propagate(mc)
      end

    else
      # Wrapping
      wrap_greens!(mc, g.greens, g.current_slice, -1)
      g.current_slice -= 1
    end
  end
  # compare(g.greens,calculate_greens_udv(p,l,g.current_slice))
  # compare(g.greens,calculate_greens_udv_chkr(p,l,g.current_slice))
  nothing
end
