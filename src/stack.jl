mutable struct MeasStack{G<:Number} # G = GreensEltype

  # ETPCs (equal-time pairing correlations)
  etpc_minus::Array{Float64, 2} # "P_(y,x)", i.e. d-wave
  etpc_plus::Array{Float64, 2} # "P+(y,x)", i.e. s-wave

  # ZFPCs (zero-frequency pairing correlations)
  zfpc_minus::Array{Float64, 2} # "P_(y,x)", i.e. d-wave
  zfpc_plus::Array{Float64, 2} # "P+(y,x)", i.e. s-wave


  # ETCDCs (equal-time charge density correlations)
  etcdc_minus::Array{Float64, 2} # "C_(y,x)", i.e. d-wave
  etcdc_plus::Array{Float64, 2} # "C+(y,x)", i.e. s-wave

  # ZFCDCs (zero-frequency charge density correlations)
  zfcdc_minus::Array{Float64, 2} # "C_(y,x)", i.e. d-wave
  zfcdc_plus::Array{Float64, 2} # "C+(y,x)", i.e. s-wave


  # ZFCCCs (zero-frequency current-current correlations)
  zfccc::Array{Float64, 2} # "Λxx(y,x)"


  # TDGF
  Gt0::Vector{Matrix{G}}
  G0t::Vector{Matrix{G}}
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

  MeasStack{G}() where G = new{G}()
end

mutable struct Stack{G<:Number} # G = GreensEltype
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

  meas::MeasStack{G}
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


  Stack{G}() where G = new{G}()
end






@def stackshortcuts begin
  s = mc.s
  N = mc.l.sites
  flv = mc.p.flv
  safe_mult = mc.p.safe_mult
  G = geltype(mc)
end


function allocate_global_update!(mc)
  @stackshortcuts
  # Global update backup
  s.gb_u_stack = zero(s.u_stack)
  s.gb_d_stack = zero(s.d_stack)
  s.gb_t_stack = zero(s.t_stack)
  s.gb_greens = zero(s.greens)
  s.gb_log_det = 0.
  s.gb_hsfield = zero(mc.p.hsfield)
end

function allocate_etgreens_stack!(mc)
  @stackshortcuts
  s.u_stack = zeros(G, flv*N, flv*N, s.n_elements)
  s.d_stack = zeros(Float64, flv*N, s.n_elements)
  s.t_stack = zeros(G, flv*N, flv*N, s.n_elements)
end

function allocate_calc_detratio!(mc)
  @stackshortcuts
  ## calc_detratio
  s.M = zeros(G, flv, flv)
  s.Mtmp = s.eye_flv - s.greens[1:N:end,1:N:end]
  s.delta_i = zeros(G, size(s.eye_flv))
  s.Mtmp2 = zeros(G, size(s.eye_flv))
  s.eVop1eVop2 = zeros(G, size(s.eye_flv))
  s.eVop1 = zeros(G, flv, flv)
  s.eVop2 = zeros(G, flv, flv)
end

function allocate_calc_greens!(mc)
  @stackshortcuts
  s.Ul = Matrix{G}(I, flv*N, flv*N)
  s.Ur = Matrix{G}(I, flv*N, flv*N)
  s.Tl = Matrix{G}(I, flv*N, flv*N)
  s.Tr = Matrix{G}(I, flv*N, flv*N)
  s.Dl = ones(Float64, flv*N)
  s.Dr = ones(Float64, flv*N)
end

function allocate_update_greens!(mc)
  @stackshortcuts
  ## update_greens
  s.A = s.greens[:,1:N:end]
  s.B = s.greens[1:N:end,:]
  s.AB = s.A * s.B
end

function allocate_propagate!(mc)
  @stackshortcuts
  s.greens_temp = zeros(G, flv*N, flv*N)
end

function _initialize_stack(mc::AbstractDQMC)
  @stackshortcuts
  s.n_elements = convert(Int, mc.p.slices / safe_mult) + 1

  s.ranges = UnitRange[]
  for i in 1:s.n_elements - 1
    push!(s.ranges, 1 + (i - 1) * safe_mult:i * safe_mult)
  end

  s.eye_flv = Matrix{Float64}(I, flv, flv)
  s.eye_full = Matrix{Float64}(I, flv*N,flv*N)
  s.ones_vec = ones(flv*N)

  s.greens = zeros(G, flv*N, flv*N)

  # interaction matrix
  s.C = zeros(G, N)
  s.S = zeros(G, N)
  s.R = zeros(G, N)
  s.eV = spzeros(G, flv*N, flv*N)

  # init empty meas stack
  s.meas = MeasStack{G}()

  # slice matrix
  cbtype(mc) === CBFalse && (s.Bl = zeros(G, flv*N, flv*N))

  # unsure about those
  s.curr_U = zeros(G, flv*N, flv*N) # used in tdgf, add_slice_sequence_ and propagate
  s.D = zeros(Float64, flv*N) # used in calc_greens in mc (below) and in calc greens in fermion meas.
  s.d = zeros(Float64, flv*N) # same as above
  s.tmp = zeros(G, flv*N, flv*N)
  s.tmp2 = zeros(G, flv*N, flv*N)
  s.U = zeros(G, flv*N, flv*N)
  s.T = zeros(G, flv*N, flv*N)
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
  s = mc.s
  a = mc.a

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
  s = mc.s
  a = mc.a

  copyto!(s.curr_U, s.u_stack[:, :, idx])

  # println("Adding slice seq left $idx = ", s.ranges[idx])
  for slice in s.ranges[idx]
    multiply_B_left!(mc, slice, s.curr_U)
  end

  @views rmul!(s.curr_U, Diagonal(s.d_stack[:, idx]))
  @mytimeit a.to "decompose_udt!" s.u_stack[:, :, idx + 1], T = @views decompose_udt!(s.curr_U, s.d_stack[:, idx + 1])
  @views mul!(s.t_stack[:, :, idx + 1],  T, s.t_stack[:, :, idx])
  nothing
end


"""
Updates stack[idx] based on stack[idx+1]
"""
function add_slice_sequence_right(mc::AbstractDQMC, idx::Int)
  s = mc.s
  a = mc.a

  copyto!(s.curr_U, s.u_stack[:, :, idx + 1])

  for slice in reverse(s.ranges[idx])
    multiply_daggered_B_left!(mc, slice, s.curr_U)
  end

  @views rmul!(s.curr_U, Diagonal(s.d_stack[:, idx + 1]))
  @mytimeit a.to "decompose_udt!" s.u_stack[:, :, idx], T = @views decompose_udt!(s.curr_U, s.d_stack[:, idx])
  @views mul!(s.t_stack[:, :, idx], T, s.t_stack[:, :, idx + 1])
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
  # TODO: Use inv_one_plus_two_udts from linalg.jl here
  s = mc.s
  tmp = mc.s.tmp
  tmp2 = mc.s.tmp2
  a = mc.a
  @mytimeit a.to "calculate_greens" begin

  mul!(tmp, s.Tl, adjoint(s.Tr))
  rmul!(tmp, Diagonal(s.Dr))
  lmul!(Diagonal(s.Dl), tmp)
  @mytimeit a.to "decompose_udt!" s.U, s.T = decompose_udt!(tmp, s.D)

  mul!(tmp, s.Ul, s.U)
  s.U .= tmp
  mul!(tmp2, s.T, adjoint(s.Ur))
  myrdiv!(tmp, s.U', tmp2) # or tmp = s.U' / tmp2 or mul!(tmp, adjoint(s.U), inv(tmp2))
  tmp[diagind(tmp)] .+= s.D
  @mytimeit a.to "decompose_udt!" u, t = decompose_udt!(tmp, s.d)

  mul!(tmp, t, tmp2)
  s.T = inv(tmp)
  mul!(tmp, s.U, u)
  s.U = adjoint(tmp)
  s.d .= 1 ./ s.d

  copyto!(tmp2, s.U)
  lmul!(Diagonal(s.d), tmp2)
  mul!(s.greens, s.T, tmp2)
  end #timeit
  nothing
end




"""
Only reasonable immediately after calculate_greens() because it depends on s.U, s.d and s.T!
"""
function calculate_logdet(mc::AbstractDQMC)
  s = mc.s

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
  s = mc.s
  p = mc.p

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
          copyto!(s.greens_temp, s.greens)
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
          copyto!(s.greens_temp, s.greens)
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
