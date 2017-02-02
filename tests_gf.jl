# Calculate B(stop) ... B(start) naively (without stabilization)
# Returns: tuple of result (matrix) and log singular values of the intermediate products
function calculate_slice_matrix_chain_naive(p::Parameters, l::Lattice, start::Int, stop::Int)
  R = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  svs = zeros(p.flv*l.sites,length(start:stop))
  svc = 1
  for k in start:stop
    R = slice_matrix_no_chkr(p,l,k) * R
    F = decompose_udv(R)
    svs[:,svc] = log(F[:S])
    svc += 1
  end
  return (R, svs)
end


# Calculate B(stop) ... B(start) safely (with stabilization at every step)
# Returns: tuple of results (U, D, and V) and log singular values of the intermediate products
function calculate_slice_matrix_chain_udv(p::Parameters, l::Lattice, start::Int, stop::Int)
  U = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  Vt = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  D = ones(Float64, p.flv*l.sites)
  svs = zeros(p.flv*l.sites,length(start:stop))
  svc = 1
  for k in start:stop
    U = slice_matrix_no_chkr(p,l,k) * U * spdiagm(D)
    F = decompose_udv!(U)
    U = F[:U]
    D = F[:S]
    Vt =  F[:Vt] * Vt
    svs[:,svc] = log(D)
    svc += 1
  end
  return (U,D,Vt,svs)
end


using PyPlot
using PyCall
@pyimport matplotlib.ticker as ticker
function plot_svs_of_slice_matrix_chain_naive(p::Parameters, l::Lattice)
  T = calculate_slice_matrix_chain_naive(p,l,1,p.slices)
  svs = T[2]
  figure()
  a = gca()
  a[:yaxis][:set_major_locator](ticker.MaxNLocator(symmetric=true))
  plot(svs[:,:]')
  ylabel("log singular values of B(\\beta,0)")
  xlabel("Inverse temperature \\beta")
  nothing
end
function plot_svs_of_slice_matrix_chain_udv(p::Parameters, l::Lattice)
  T = calculate_slice_matrix_chain_udv(p,l,1,p.slices)
  svs = T[4]
  figure()
  a = gca()
  a[:yaxis][:set_major_locator](ticker.MaxNLocator(symmetric=true))
  plot(svs[:,:]')
  ylabel("log singular values of B(\\beta,0)")
  xlabel("Inverse temperature \\beta")
  println(maximum(svs))
  nothing
end
function plot_svs_of_slice_matrix_chain_both(p::Parameters, l::Lattice)
  T = calculate_slice_matrix_chain_naive(p,l,1,p.slices)
  svs = T[2]
  fig = figure()
  b = fig[:add_subplot](121)
  b[:yaxis][:set_major_locator](ticker.MaxNLocator(symmetric=true))
  b[:plot](svs[:,:]')
  b[:set_xlabel]("Inverse temperature \$\\beta\$")
  b[:set_ylabel]("log singular values of \$B(\\beta,0)\$")

  T = calculate_slice_matrix_chain_udv(p,l,1,p.slices)
  svs = T[4]
  a = fig[:add_subplot](122, sharey=b, sharex=b)
  a[:plot](svs[:,:]')
  setp(a[:get_yticklabels](), visible=false)
  a[:set_xlabel]("Inverse temperature \$\\beta\$")

  # tight_layout()
  subplots_adjust(wspace = 0.0)
  println(maximum(svs))
  nothing
end





"""
Calculate G(slice) = [1+B(slice-1)...B(1)B(M) ... B(slice)]^(-1) using
udv decompositions for (!)every slice matrix multiplication
"""
function calculate_greens_udv(p::Parameters, l::Lattice, slice::Int)
  # Calculate Ur,Dr,Vtr=B(M) ... B(slice)
  Ur, Dr, Vtr = calculate_slice_matrix_chain_udv(p,l,slice,p.slices)

  # Calculate Ul,Dl,Vtl=B(slice-1) ... B(1)
  Ul, Dl, Vtl = calculate_slice_matrix_chain_udv(p,l,1,slice-1)

  # Calculate Greens function
  tmp = Vtl * Ur
  inner = ctranspose(Vtr * Ul) + spdiagm(Dl) * tmp * spdiagm(Dr)
  I = decompose_udv!(inner)
  return ctranspose(I[:Vt] * Vtr) * spdiagm(1./I[:S]) * ctranspose(Ul * I[:U])
end


function calculate_greens_naive(p::Parameters, l::Lattice, slice::Int)
  # Calculate Ur,Dr,Vtr=B(M) ... B(slice)
  Br = calculate_slice_matrix_chain_naive(p,l,slice,p.slices)[1]

  # Calculate Ul,Dl,Vtl=B(slice-1) ... B(1)
  Bl = calculate_slice_matrix_chain_naive(p,l,1,slice-1)[1]

  # Calculate Greens function
  invgreens = eye(p.flv*l.sites)+ Bl * Br
  return inv(invgreens)
end


function calculate_greens_naive_udvinv(p::Parameters, l::Lattice, slice::Int)
  # Calculate Ur,Dr,Vtr=B(M) ... B(slice)
  Br = calculate_slice_matrix_chain_naive(p,l,slice,p.slices)[1]
  F = decompose_udv!(Br)
  Ur = F[:U]
  Dr = F[:S]
  Vtr = F[:Vt]

  # Calculate Ul,Dl,Vtl=B(slice-1) ... B(1)
  Bl = calculate_slice_matrix_chain_naive(p,l,1,slice-1)[1]
  F = decompose_udv!(Bl)
  Ul = F[:U]
  Dl = F[:S]
  Vtl = F[:Vt]

  # Calculate Greens function
  tmp = Vtl * Ur
  inner = ctranspose(Vtr * Ul) + spdiagm(Dl) * tmp * spdiagm(Dr)
  I = decompose_udv!(inner)
  return ctranspose(I[:Vt] * Vtr) * spdiagm(1./I[:S]) * ctranspose(Ul * I[:U])
end


function compare_greens(g1::Array{Complex{Float64}, 2}, g2::Array{Complex{Float64}, 2})
  println("max dev: ", maximum(abs(g1 - g2)))
  println("mean dev: ", mean(abs(g1 - g2)))
  println("max rel dev: ", maximum(reldiff(g1,g2)))
  println("mean rel dev: ", mean(reldiff(g1,g2)))
  println("max diag dev: ", maximum(diag(abs(g1 - g2))))
  println("mean diag dev: ", mean(diag(abs(g1 - g2))))
  println("max diag rel dev: ", maximum(diag(reldiff(g1,g2))))
  println("mean diag rel dev: ", mean(diag(reldiff(g1,g2))))
  return isapprox(g1,g2,atol=1e-2)
end

function check_current_gf(s,p,l)
  gf = calculate_greens_udv(p,l,s.current_slice)
  compare_greens(gf,s.greens)
end



function test_greens_udv_vs_naive_vs_naiveudvinv(p::Parameters, l::Lattice)
  slice = rand(1:p.slices)
  gfudv = calculate_greens_udv(p,l,slice)
  gfnaive = calculate_greens_naive(p,l,slice)
  gfnaiveudvinv = calculate_greens_naive_udvinv(p,l,slice)
  if compare_greens(gfnaive,gfudv)
    error("Naive Green's function equals stabilized Green's function! Very small number of slices?")
  elseif !compare_greens(gfnaive,gfnaiveudvinv)
    error("Inversion process in naive Green's function calculation seems to matter!")
  end
  return true
end
# Worked.


function wrap_greens(gf::Array{Complex{Float64},2},slice::Int,direction::Int)
  if direction == -1
    A = slice_matrix_no_chkr(p, l, slice - 1, -1.)
    temp = A * gf
    A = slice_matrix_no_chkr(p, l, slice - 1, 1.)
    return temp * A
  else
    A = slice_matrix_no_chkr(p, l, slice, 1.)
    temp = A * gf
    A = slice_matrix_no_chkr(p, l, slice, -1.)
    return temp * A
  end
end



function test_gf_wrapping(p::Parameters, l::Lattice)
  slice = rand(1:p.slices)
  gf = calculate_greens_udv(p,l,slice)

  gfwrapped = wrap_greens(gf,slice,-1)
  gfwrapped2 = wrap_greens(gfwrapped,slice,-1)
  gfexact = calculate_greens_udv(p,l,slice - 1)
  gfexact2 = calculate_greens_udv(p,l,slice - 2)
  gfexact3 = calculate_greens_udv(p,l,slice - 2)

  println("Comparing wrapped vs num exact")
  compare_greens(gfwrapped,gfexact)
  println("")
  println("Comparing twice wrapped vs num exact")
  compare_greens(gfwrapped2,gfexact2)
  nothing
end


function test_local_update_gf(s::Stack, p::Parameters, l::Lattice)
  p.hsfield = rand(3,l.sites,p.slices)
  s.current_slice = rand(1:p.slices)

  gfbefore = calculate_greens_udv(p,l,slice)
  s.greens = copy(gf)

  site = rand(1:l.sites)
  new_op = rand(p.box, 3)
  detratio = calculate_detratio(s,p,l,site,new_op)
  update_greens!(s,p,l,site)
  p.hsfield[:,site,s.current_slice] = new_op[:]

  gfafter = calculate_greens_udv(p,l,slice)

  println("Num exact vs updated")
  compare_greens(s.greens, gfafter)
  println("")
  println("Before vs After (num exact)")
  compare_greens(gfbefore, gfafter)
  nothing
end



function test_gf_safe_mult(s,p,l)
  p.slices = 200
  p.safe_mult = p.slices
  p.hsfield = rand(3,l.sites,p.slices)
  initialize_stack(s,p,l)
  build_stack(s,p,l)
  propagate(s,p,l)
  if check_current_gf(s,p,l)
    error("p.safe_mult = p.slices and gf is still correct!?!")
  end

  p.slices = 200
  p.safe_mult = 1
  p.hsfield = rand(3,l.sites,p.slices)

  p.safe_mult = 1
  initialize_stack(s,p,l)
  build_stack(s,p,l)
  propagate(s,p,l)
  if !check_current_gf(s,p,l)
    error("p.safe_mult=1 and gf is not correct!?!")
  end
end
