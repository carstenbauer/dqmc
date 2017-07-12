"""
Calculate slice matrix chain
"""
# Calculate B(stop) ... B(start) naively (without stabilization)
# Returns: tuple of result (matrix) and log singular values of the intermediate products
function calculate_slice_matrix_chain_naive(p::Parameters, l::Lattice, start::Int, stop::Int)
  assert(0 < start <= p.slices)
  assert(0 < stop <= p.slices)
  assert(start <= stop)

  R = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  U = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  D = ones(Float64, p.flv*l.sites)
  Vt = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)

  svs = zeros(p.flv*l.sites,length(start:stop))
  svc = 1
  for k in start:stop
    R = slice_matrix_no_chkr(p,l,k) * R
    U, D, Vt = decompose_udv(R)
    svs[:,svc] = log.(D)
    svc += 1
  end
  return (R, svs)
end


# Calculate B(stop) ... B(start) safely (with stabilization at every safe_mult step, default ALWAYS)
# Returns: tuple of results (U, D, and V) and log singular values of the intermediate products
function calculate_slice_matrix_chain_udv(p::Parameters, l::Lattice, start::Int, stop::Int, safe_mult::Int=1)
  # assert(0 < start <= p.slices)
  # assert(0 < stop <= p.slices)
  # assert(start <= stop)

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
      svs[:,svc] = log.(D)
      svc += 1
    else
      U = slice_matrix_no_chkr(p,l,k) * U
    end
  end
  return (U,D,Vt,svs)
end


function calculate_slice_matrix_chain_udv_chkr(p::Parameters, l::Lattice, start::Int, stop::Int, safe_mult::Int=1)
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
      svs[:,svc] = log.(D)
      svc += 1
    else
      U = slice_matrix(p,l,k) * U
    end
  end
  return (U,D,Vt,svs)
end

# Calculate Ul, Dl, Tl =B(stop) ... B(start)
function calculate_slice_matrix_chain_qr(p::Parameters, l::Lattice, start::Int, stop::Int, safe_mult::Int=1)
  assert(0 < start <= p.slices)
  assert(0 < stop <= p.slices)
  assert(start <= stop)

  U = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  D = ones(Float64, p.flv*l.sites)
  T = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  Tnew = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)

  svs = zeros(p.flv*l.sites,length(start:stop))
  svc = 1
  for k in start:stop
    if mod(k,safe_mult) == 0
      U = slice_matrix_no_chkr(p,l,k) * U * spdiagm(D)
      U, D, Tnew = decompose_udt(U)
      T =  Tnew * T
      svs[:,svc] = log.(D)
      svc += 1
    else
      U = slice_matrix_no_chkr(p,l,k) * U
    end
  end
  return (U,D,T,svs)
end

# Calculate (Ur, Dr, Tr)' = B(stop) ... B(start)  => Ur,Dr, Tr = B(start)' ... B(stop)'
function calculate_slice_matrix_chain_qr_dagger(p::Parameters, l::Lattice, start::Int, stop::Int, safe_mult::Int=1)
  assert(0 < start <= p.slices)
  assert(0 < stop <= p.slices)
  assert(start <= stop)

  U = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  D = ones(Float64, p.flv*l.sites)
  T = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  Tnew = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)

  svs = zeros(p.flv*l.sites,length(start:stop))
  svc = 1
  for k in reverse(start:stop)
    if mod(k,safe_mult) == 0
      U = ctranspose(slice_matrix_no_chkr(p,l,k)) * U * spdiagm(D)
      U, D, Tnew = decompose_udt(U)
      T =  Tnew * T
      svs[:,svc] = log.(D)
      svc += 1
    else
      U = ctranspose(slice_matrix_no_chkr(p,l,k)) * U
    end
  end
  return (U,D,T,svs)
end


"""
Calculate Green's function (direct, i.e. without stack)
"""
# Calculate G(slice) = [1+B(slice-1)...B(1)B(M) ... B(slice)]^(-1) using
# udv decompositions for (!)every slice matrix multiplication
function calculate_greens_udv(p::Parameters, l::Lattice, slice::Int)
  # Calculate Ur,Dr,Vtr=B(M) ... B(slice)
  Ur, Dr, Vtr = calculate_slice_matrix_chain_udv(p,l,slice,p.slices)

  # Calculate Ul,Dl,Vtl=B(slice-1) ... B(1)
  if slice-1 >= 1
    Ul, Dl, Vtl = calculate_slice_matrix_chain_udv(p,l,1,slice-1)
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

function calculate_greens_udv_chkr(p::Parameters, l::Lattice, slice::Int)
  # Calculate Ur,Dr,Vtr=B(M) ... B(slice)
  Ur, Dr, Vtr = calculate_slice_matrix_chain_udv_chkr(p,l,slice,p.slices)

  # Calculate Ul,Dl,Vtl=B(slice-1) ... B(1)
  if slice-1 >= 1
    Ul, Dl, Vtl = calculate_slice_matrix_chain_udv_chkr(p,l,1,slice-1)
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

function calculate_greens_2udv(p::Parameters, l::Lattice, slice::Int)
  # Calculate Ur,Dr,Vtr=B(M) ... B(slice)
  Ur, Dr, Vtr = calculate_slice_matrix_chain_udv(p,l,slice,p.slices)

  # Calculate Ul,Dl,Vtl=B(slice-1) ... B(1)
  if slice-1 >= 1
    Ul, Dl, Vtl = calculate_slice_matrix_chain_udv(p,l,1,slice-1)
  else
    Ul = eye(Complex128, p.flv * l.sites)
    Dl = ones(Float64, p.flv * l.sites)
    Vtl = eye(Complex128, p.flv * l.sites)
  end

  # Calculate Greens function
  tmp = Vtl * Ur
  inner = spdiagm(Dl) * tmp * spdiagm(Dr)
  I = decompose_udv!(inner)

  U = Ul*I[1]
  D = spdiagm(I[2])
  Vt = I[3] * Vtr

  F = decompose_udv!(ctranspose(Vt*U) + D)

  return ctranspose(F[3] * Vt) * spdiagm(1./F[2]) * ctranspose(U * F[1])
end
# equivalent to version above with only one SVD decomp. up to max absdiff of ~1e-10


function calculate_greens_naive(p::Parameters, l::Lattice, slice::Int)
  # Calculate Ur,Dr,Vtr=B(M) ... B(slice)
  Br = calculate_slice_matrix_chain_naive(p,l,slice,p.slices)[1]

  # Calculate Ul,Dl,Vtl=B(slice-1) ... B(1)
  if slice-1 >= 1
    Bl = calculate_slice_matrix_chain_naive(p,l,1,slice-1)[1]
  else
    Bl = eye(Complex128, p.flv * l.sites)
  end

  # Calculate Greens function
  invgreens = eye(p.flv*l.sites)+ Bl * Br
  return inv(invgreens)
end


function calculate_greens_naive_udvinv(p::Parameters, l::Lattice, slice::Int)
  # Calculate Ur,Dr,Vtr=B(M) ... B(slice)
  Br = calculate_slice_matrix_chain_naive(p,l,slice,p.slices)[1]
  Ur, Dr, Vtr = decompose_udv!(Br)

  # Calculate Ul,Dl,Vtl=B(slice-1) ... B(1)
  if slice-1 >= 1
    Bl = calculate_slice_matrix_chain_naive(p,l,1,slice-1)[1]
  else
    Bl = eye(Complex128, p.flv * l.sites)
  end
  Ul, Dl, Vtl = decompose_udv!(Bl)

  # Calculate Greens function
  tmp = Vtl * Ur
  inner = ctranspose(Vtr * Ul) + spdiagm(Dl) * tmp * spdiagm(Dr)
  I = decompose_udv!(inner)
  return ctranspose(I[3] * Vtr) * spdiagm(1./I[2]) * ctranspose(Ul * I[1])
end

function calculate_greens_qr(p::Parameters, l::Lattice, slice::Int)
  # Calculate Ur,Dr,Tr=B(slice)' ... B(M)'
  Ur, Dr, Tr = calculate_slice_matrix_chain_qr_dagger(p,l,slice,p.slices)

  # Calculate Ul,Dl,Tl=B(slice-1) ... B(1)
  if slice-1 >= 1
    Ul, Dl, Tl = calculate_slice_matrix_chain_qr(p,l,1,slice-1)
  else
    Ul = eye(Complex128, p.flv * l.sites)
    Dl = ones(Float64, p.flv * l.sites)
    Tl = eye(Complex128, p.flv * l.sites)
  end

  tmp = Tl * ctranspose(Tr)
  U, D, T = decompose_udt(spdiagm(Dl) * tmp * spdiagm(Dr))
  U = Ul * U
  T *= ctranspose(Ur)

  u, d, t = decompose_udt(/(ctranspose(U), T) + spdiagm(D))

  ldet = 1/(prod(d))

  return \(t*T,spdiagm(1./d)*ctranspose(U * u))

end


"""
Helpers
"""
function compare_greens(g1::Array{Complex{Float64}, 2}, g2::Array{Complex{Float64}, 2})
  println("max dev: ", maximum(absdiff(g1,g2)))
  println("mean dev: ", mean(absdiff(g1,g2)))
  println("max rel dev: ", maximum(reldiff(g1,g2)))
  println("mean rel dev: ", mean(reldiff(g1,g2)))
  return isapprox(g1,g2,atol=1e-2,rtol=1e-1)
end

function check_current_gf(s,p,l)
  gf = calculate_greens_udv(p,l,s.current_slice)
  compare_greens(gf,s.greens)
end
