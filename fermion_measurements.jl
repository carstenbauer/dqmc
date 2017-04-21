"""
Measurements
"""
function measure_greens_and_det(p::Parameters, l::Lattice, safe_mult::Int=1)
  greens, greens_svs = calculate_greens_and_svs_chkr(p, l, 1, safe_mult)
  return effective_greens2greens!(p, l, greens), prod(greens_svs)
end

function measure_greens_and_det_no_chkr(p::Parameters, l::Lattice, safe_mult::Int=1)
  greens, greens_svs = calculate_greens_and_svs(p, l, 1, safe_mult)
  return effective_greens2greens_no_chkr!(p, l, greens), prod(greens_svs)
end


"""
Effective Green's function -> Green's function
"""
function effective_greens2greens!(p::Parameters, l::Lattice, greens::Array{Complex{Float64}, 2})
  greens[:] = greens * l.chkr_hop_half[2]
  greens[:] = greens * l.chkr_hop_half[1]
  greens[:] = l.chkr_hop_half_inv[2] * greens
  greens[:] = l.chkr_hop_half_inv[1] * greens
end

function effective_greens2greens(p::Parameters, l::Lattice, greens::Array{Complex{Float64}, 2})
  g = copy(greens)
  effective_greens2greens!(p, l, g)
  return g
end

function effective_greens2greens_no_chkr!(p::Parameters, l::Lattice, greens::Array{Complex{Float64}, 2})
  greens[:] = greens * l.hopping_matrix_exp
  greens[:] = l.hopping_matrix_exp * greens
end


"""
Calculate effective(!) Green's function (direct, i.e. without stack)
"""
# Calculate B(stop) ... B(start) safely (with stabilization at every safe_mult step, default ALWAYS)
# Returns: tuple of results (U, D, and V) and log singular values of the intermediate products
function calculate_slice_matrix_chain(p::Parameters, l::Lattice, start::Int, stop::Int, safe_mult::Int=1)
  assert(0 < start <= p.slices)
  assert(0 < stop <= p.slices)
  assert(start <= stop)

  U = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  D = ones(Float64, p.flv*l.sites)
  Vt = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  Vtnew = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)

  svs = zeros(p.flv*l.sites,length(start:stop))
  svc = 1
  for k in start:stop
    if mod(k,safe_mult) == 0
      U = slice_matrix_no_chkr(p,l,k) * U * spdiagm(D)
      U, D, Vtnew = decompose_udv!(U)
      Vt =  Vtnew * Vt
      svs[:,svc] = log(D)
      svc += 1
    else
      U = slice_matrix_no_chkr(p,l,k) * U
    end
  end
  return (U,D,Vt,svs)
end

function calculate_slice_matrix_chain_chkr(p::Parameters, l::Lattice, start::Int, stop::Int, safe_mult::Int=1)
  assert(0 < start <= p.slices)
  assert(0 < stop <= p.slices)
  assert(start <= stop)

  U = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  D = ones(Float64, p.flv*l.sites)
  Vt = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  Vtnew = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)

  svs = zeros(p.flv*l.sites,length(start:stop))
  svc = 1
  for k in start:stop
    if mod(k,safe_mult) == 0
      U = slice_matrix(p,l,k) * U * spdiagm(D)
      U, D, Vtnew = decompose_udv(U)
      Vt =  Vtnew * Vt
      svs[:,svc] = log(D)
      svc += 1
    else
      U = slice_matrix(p,l,k) * U
    end
  end
  return (U,D,Vt,svs)
end

# Calculate G(slice) = [1+B(slice-1)...B(1)B(M) ... B(slice)]^(-1) in a stable manner
function calculate_greens(p::Parameters, l::Lattice, slice::Int, safe_mult::Int=1)
  # Calculate Ur,Dr,Vtr=B(M) ... B(slice)
  Ur, Dr, Vtr = calculate_slice_matrix_chain(p,l,slice,p.slices,safe_mult)

  # Calculate Ul,Dl,Vtl=B(slice-1) ... B(1)
  if slice-1 >= 1
    Ul, Dl, Vtl = calculate_slice_matrix_chain(p,l,1,slice-1,safe_mult)
  else
    Ul = eye(Complex128, p.flv * l.sites)
    Dl = ones(Float64, p.flv * l.sites)
    Vtl = eye(Complex128, p.flv * l.sites)
  end

  # Calculate Greens function
  tmp = Vtl * Ur
  inner = ctranspose(Vtr * Ul) + spdiagm(Dl) * tmp * spdiagm(Dr)
  I = decompose_udv!(inner)
  return ctranspose(I[3] * Vtr) * spdiagm(1./I[2]) * ctranspose(Ul * I[1])
end

function calculate_greens_chkr(p::Parameters, l::Lattice, slice::Int, safe_mult::Int=1)
  # Calculate Ur,Dr,Vtr=B(M) ... B(slice)
  Ur, Dr, Vtr = calculate_slice_matrix_chain_chkr(p,l,slice,p.slices,safe_mult)

  # Calculate Ul,Dl,Vtl=B(slice-1) ... B(1)
  if slice-1 >= 1
    Ul, Dl, Vtl = calculate_slice_matrix_chain_chkr(p,l,1,slice-1,safe_mult)
  else
    Ul = eye(Complex128, p.flv * l.sites)
    Dl = ones(Float64, p.flv * l.sites)
    Vtl = eye(Complex128, p.flv * l.sites)
  end

  # Calculate Greens function
  tmp = Vtl * Ur
  inner = ctranspose(Vtr * Ul) + spdiagm(Dl) * tmp * spdiagm(Dr)
  I = decompose_udv!(inner)
  return ctranspose(I[3] * Vtr) * spdiagm(1./I[2]) * ctranspose(Ul * I[1])
end

# Calculate G(slice) = [1+B(slice-1)...B(1)B(M) ... B(slice)]^(-1) and its singular values in a stable manner
function calculate_greens_and_svs(p::Parameters, l::Lattice, slice::Int, safe_mult::Int=1)
  # Calculate Ur,Dr,Vtr=B(M) ... B(slice)
  Ur, Dr, Vtr = calculate_slice_matrix_chain(p,l,slice,p.slices,safe_mult)

  # Calculate Ul,Dl,Vtl=B(slice-1) ... B(1)
  if slice-1 >= 1
    Ul, Dl, Vtl = calculate_slice_matrix_chain(p,l,1,slice-1,safe_mult)
  else
    Ul = eye(Complex128, p.flv * l.sites)
    Dl = ones(Float64, p.flv * l.sites)
    Vtl = eye(Complex128, p.flv * l.sites)
  end

  # Calculate Greens function
  tmp = Vtl * Ur
  inner = ctranspose(Vtr * Ul) + spdiagm(Dl) * tmp * spdiagm(Dr)
  I = decompose_udv!(inner)
  U = ctranspose(I[3] * Vtr)
  D = spdiagm(1./I[2])
  Vt = ctranspose(Ul * I[1])
  return (U*D*Vt, diag(D))
end

function calculate_greens_and_svs_chkr(p::Parameters, l::Lattice, slice::Int, safe_mult::Int=1)
  # Calculate Ur,Dr,Vtr=B(M) ... B(slice)
  Ur, Dr, Vtr = calculate_slice_matrix_chain_chkr(p,l,slice,p.slices,safe_mult)

  # Calculate Ul,Dl,Vtl=B(slice-1) ... B(1)
  if slice-1 >= 1
    Ul, Dl, Vtl = calculate_slice_matrix_chain_chkr(p,l,1,slice-1,safe_mult)
  else
    Ul = eye(Complex128, p.flv * l.sites)
    Dl = ones(Float64, p.flv * l.sites)
    Vtl = eye(Complex128, p.flv * l.sites)
  end

  # Calculate Greens function
  tmp = Vtl * Ur
  inner = ctranspose(Vtr * Ul) + spdiagm(Dl) * tmp * spdiagm(Dr)
  I = decompose_udv!(inner)
  U = ctranspose(I[3] * Vtr)
  D = spdiagm(1./I[2])
  Vt = ctranspose(Ul * I[1])
  return (U*D*Vt, diag(D))
end


"""
Helpers
"""
# log(det(G)) = S_F
# log(det(G^-1)) = -S_F
function det2fermion_action(greens_det::Float64)
  return log(greens_det)
end