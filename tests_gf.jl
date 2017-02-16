# Calculate B(stop) ... B(start) naively (without stabilization)
# Returns: tuple of result (matrix) and log singular values of the intermediate products
function calculate_slice_matrix_chain_naive(p::Parameters, l::Lattice, start::Int, stop::Int)
  R = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  U = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)
  D = ones(Float64, p.flv*l.sites)
  Vt = eye(Complex{Float64}, p.flv*l.sites, p.flv*l.sites)

  svs = zeros(p.flv*l.sites,length(start:stop))
  svc = 1
  for k in start:stop
    R = slice_matrix_no_chkr(p,l,k) * R
    U, D, Vt = decompose_udv(R)
    svs[:,svc] = log(D)
    svc += 1
  end
  return (R, svs)
end


# Calculate B(stop) ... B(start) safely (with stabilization at every safe_mult step, default ALWAYS)
# Returns: tuple of results (U, D, and V) and log singular values of the intermediate products
function calculate_slice_matrix_chain_udv(p::Parameters, l::Lattice, start::Int, stop::Int, safe_mult::Int=1)
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


function calculate_slice_matrix_chain_udv_chkr(p::Parameters, l::Lattice, start::Int, stop::Int, safe_mult::Int=1)
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


using PyPlot
using PyCall
@pyimport matplotlib.ticker as ticker
function plot_svs_of_slice_matrix_chain(p::Parameters, l::Lattice)
  T = calculate_slice_matrix_chain_naive(p,l,1,p.slices)
  svs = T[2]
  fig = figure(figsize=(20,7))
  b = fig[:add_subplot](131)
  b[:yaxis][:set_major_locator](ticker.MaxNLocator(symmetric=true))
  b[:plot](svs[:,:]')
  b[:set_xlabel]("Inverse temperature \$\\beta\$")
  b[:set_ylabel]("log singular values of \$B(\\beta,0)\$")

  T = calculate_slice_matrix_chain_udv(p,l,1,p.slices,1)
  svs = T[4]
  a = fig[:add_subplot](132, sharey=b, sharex=b)
  a[:plot](svs[:,:]')
  setp(a[:get_yticklabels](), visible=false)
  a[:set_xlabel]("Inverse temperature \$\\beta\$")

  T = calculate_slice_matrix_chain_udv_chkr(p,l,1,p.slices)
  svs = T[4]
  c = fig[:add_subplot](133, sharey=b, sharex=b)
  c[:plot](svs[:,:]')
  setp(c[:get_yticklabels](), visible=false)
  c[:set_xlabel]("Inverse temperature \$\\beta\$")

  # tight_layout()
  subplots_adjust(wspace = 0.0)
  println(maximum(svs))
  nothing
end
plot_svs_of_slice_matrix_chain(p,l)


function plot_lowest_sv_of_slice_matrix_chain_vs_safe_mult(p::Parameters, l::Lattice)

  svs = Vector{Float64}(50)
  for safe_mult in 1:50
    U, D, Vt, svals = calculate_slice_matrix_chain_udv(p,l,1,p.slices,safe_mult)
    U = U * spdiagm(D)
    F = decompose_udv!(U)
    D = F[2]
    svs[safe_mult] = log(D[end])
  end

  fig, ax1 = subplots()

  # These are in unitless percentages of the figure size. (0,0 is bottom left)
  left, bottom, width, height = [0.3, 0.3, 0.2, 0.2]
  ax2 = fig[:add_axes]([left, bottom, width, height])

  ax1[:plot](collect(1:50),abs(svs[:]-svs[1]),"C0")
  ax1[:set_ylabel]("deviation of smallest log singular value")
  ax1[:set_xlabel]("safe_mult")
  ax2[:plot](collect(1:20),abs(svs[1:20]-svs[1]),"C1")

  show()
  nothing
end
plot_lowest_sv_of_slice_matrix_chain_vs_safe_mult(p,l)




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
  return ctranspose(I[3] * Vtr) * spdiagm(1./I[2]) * ctranspose(Ul * I[1])
end

function calculate_greens_udv_chkr(p::Parameters, l::Lattice, slice::Int)
  # Calculate Ur,Dr,Vtr=B(M) ... B(slice)
  Ur, Dr, Vtr = calculate_slice_matrix_chain_udv_chkr(p,l,slice,p.slices)

  # Calculate Ul,Dl,Vtl=B(slice-1) ... B(1)
  Ul, Dl, Vtl = calculate_slice_matrix_chain_udv_chkr(p,l,1,slice-1)

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
  Ul, Dl, Vtl = calculate_slice_matrix_chain_udv(p,l,1,slice-1)

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
  Bl = calculate_slice_matrix_chain_naive(p,l,1,slice-1)[1]

  # Calculate Greens function
  invgreens = eye(p.flv*l.sites)+ Bl * Br
  return inv(invgreens)
end


function calculate_greens_naive_udvinv(p::Parameters, l::Lattice, slice::Int)
  # Calculate Ur,Dr,Vtr=B(M) ... B(slice)
  Br = calculate_slice_matrix_chain_naive(p,l,slice,p.slices)[1]
  Ur, Dr, Vtr = decompose_udv!(Br)

  # Calculate Ul,Dl,Vtl=B(slice-1) ... B(1)
  Bl = calculate_slice_matrix_chain_naive(p,l,1,slice-1)[1]
  Ul, Dl, Vtl = decompose_udv!(Bl)

  # Calculate Greens function
  tmp = Vtl * Ur
  inner = ctranspose(Vtr * Ul) + spdiagm(Dl) * tmp * spdiagm(Dr)
  I = decompose_udv!(inner)
  return ctranspose(I[3] * Vtr) * spdiagm(1./I[2]) * ctranspose(Ul * I[1])
end


function compare_greens(g1::Array{Complex{Float64}, 2}, g2::Array{Complex{Float64}, 2})
  println("max dev: ", maximum(abs(g1 - g2)))
  println("mean dev: ", mean(abs(g1 - g2)))
  println("max rel dev: ", maximum(reldiff(g1,g2)))
  println("mean rel dev: ", mean(reldiff(g1,g2)))
  # println("max diag dev: ", maximum(diag(abs(g1 - g2))))
  # println("mean diag dev: ", mean(diag(abs(g1 - g2))))
  # println("max diag rel dev: ", maximum(diag(reldiff(g1,g2))))
  # println("mean diag rel dev: ", mean(diag(reldiff(g1,g2))))
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

function wrap_greens2(gf::Array{Complex{Float64},2},slice::Int,direction::Int)
  if direction == -1
    B = slice_matrix_no_chkr(p, l, slice - 1)
    temp = inv(B) * gf
    return temp * B
  else
    B = slice_matrix_no_chkr(p, l, slice)
    temp = B * gf
    return temp * inv(B)
  end
end

function test_wrap_greens_vs_wrap_greens2(p,l)
  # slice = rand(1:p.slices)
  slice = 10
  gf = calculate_greens_udv(p,l,slice)

  # wrapping down
  gfwrapped1 = wrap_greens(gf,slice,-1)
  gfwrapped2 = wrap_greens2(gf,slice,-1)

  println("Comparing wrapped down vs num exact")
  compare_greens(gfwrapped1,gfwrapped2)
end
# Worked! Found mistake in slice_matrix_no_chkr in stack.jl



function test_gf_wrapping(p::Parameters, l::Lattice)
  # slice = rand(1:p.slices)
  slice = 10
  gf = calculate_greens_udv(p,l,slice)

  # wrapping down
  gfwrapped = wrap_greens(gf,slice,-1)
  gfwrapped2 = wrap_greens(gfwrapped,slice,-1)
  gfexact = calculate_greens_udv(p,l,slice - 1)
  gfexact2 = calculate_greens_udv(p,l,slice - 2)

  println("Comparing wrapped down vs num exact")
  compare_greens(gfwrapped,gfexact)
  println("")
  println("Comparing twice wrapped down vs num exact")
  compare_greens(gfwrapped2,gfexact2)

  # wrapping up
  gfwrapped = wrap_greens(gf,slice,1)
  gfwrapped2 = wrap_greens(gfwrapped,slice,1)
  gfexact = calculate_greens_udv(p,l,slice + 1)
  gfexact2 = calculate_greens_udv(p,l,slice + 2)

  println("")
  println("")
  println("Comparing wrapped down vs num exact")
  compare_greens(gfwrapped,gfexact)
  println("")
  println("Comparing twice wrapped down vs num exact")
  compare_greens(gfwrapped2,gfexact2)

  nothing
end
# Single wrapping: Fine. Twice wrapping: deviation to large?


function test_local_update_gf(s::Stack, p::Parameters, l::Lattice)
  p.hsfield = rand(3,l.sites,p.slices)
  s.current_slice = rand(1:p.slices)

  gfbefore = calculate_greens_udv(p,l,s.current_slice)
  s.greens = copy(gfbefore)

  for site in 1:l.sites
    new_op = rand(p.box, 3)
    detratio = calculate_detratio(s,p,l,site,new_op)
    update_greens!(s,p,l,site)
    p.hsfield[:,site,s.current_slice] = new_op[:]
  end

  gfafter = calculate_greens_udv(p,l,s.current_slice)

  println("Num exact vs updated")
  compare_greens(s.greens, gfafter)
  # println("")
  # println("Before vs After (num exact)")
  # compare_greens(gfbefore, gfafter)
  nothing
end
# What about max rel dev (order 1)? Max abs dev around 1e-3 to 1e-2


function test_gf_safe_mult(s::Stack, p::Parameters, l::Lattice)
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
# Makes sense. However, agreement only up to 1e-2.
# p.safe_mult=1 is not completely equivalent to direct implementation.


function plot_gf_error_propagation(s::Stack, p::Parameters, l::Lattice)
  p.slices = 200
  p.safe_mult = 1
  p.hsfield = rand(3,l.sites,p.slices)
  initialize_stack(s,p,l)
  build_stack(s,p,l)
  mean_dev = Vector{Float64}(p.slices)
  slices = Vector{Int}(p.slices)
  for n in 1:p.slices
    propagate(s,p,l)
    gf = calculate_greens_udv(p,l,s.current_slice)
    mean_dev[n] = mean(absdiff(gf,s.greens))
    slices[n] = s.current_slice
  end

  mean_devUP = Vector{Float64}(p.slices)
  slicesUP = Vector{Int}(p.slices)
  for n in 1:p.slices
    propagate(s,p,l)
    gf = calculate_greens_udv(p,l,s.current_slice)
    mean_devUP[n] = mean(absdiff(gf,s.greens))
    slicesUP[n] = s.current_slice
  end

  # second round (more or less copy paste of above)
  mean_dev2 = Vector{Float64}(p.slices)
  slices = Vector{Int}(p.slices)
  for n in 1:p.slices
    propagate(s,p,l)
    gf = calculate_greens_udv(p,l,s.current_slice)
    mean_dev2[n] = mean(absdiff(gf,s.greens))
    slices[n] = s.current_slice
  end

  mean_dev2UP = Vector{Float64}(p.slices)
  slicesUP = Vector{Int}(p.slices)
  for n in 1:p.slices
    propagate(s,p,l)
    gf = calculate_greens_udv(p,l,s.current_slice)
    mean_dev2UP[n] = mean(absdiff(gf,s.greens))
    slicesUP[n] = s.current_slice
  end

  fig, (ax1, ax2, ax3, ax4) = subplots(2,2,sharey=true)
  ax1[:plot](slices,mean_dev,"C0")
  ax1[:invert_xaxis]()
  ax3[:plot](slicesUP,mean_devUP,"C0")
  ax2[:plot](slices,mean_dev2,"C3")
  ax2[:invert_xaxis]()
  ax4[:plot](slicesUP,mean_dev2UP,"C3")
  subplots_adjust(wspace = 0.0)
  fig[:text](.03, .5, "mean abs error", ha="center", va="center", rotation="vertical")
  fig[:text](.5, .03, "time slice", ha="center")
  show()
end


function calculate_greens_and_det_and_svs_udv(p::Parameters, l::Lattice, slice::Int)
  # Calculate Ur,Dr,Vtr=B(M) ... B(slice)
  Ur, Dr, Vtr = calculate_slice_matrix_chain_udv(p,l,slice,p.slices)

  # Calculate Ul,Dl,Vtl=B(slice-1) ... B(1)
  Ul, Dl, Vtl = calculate_slice_matrix_chain_udv(p,l,1,slice-1)

  # Calculate Greens function
  tmp = Vtl * Ur
  inner = ctranspose(Vtr * Ul) + spdiagm(Dl) * tmp * spdiagm(Dr)
  I = decompose_udv!(inner)
  U = ctranspose(I[3] * Vtr)
  D = spdiagm(1./I[2])
  Vt = ctranspose(Ul * I[1])
  return (U*D*Vt, det(U)*det(D)*det(Vt), diag(D))
end


function test_greens_det_naive(s::Stack, p::Parameters, l::Lattice)
  greens, det_udv, svs_udv = calculate_greens_and_det_and_svs_udv(p,l,s.current_slice)
  det_naive = det(greens)
  F = decompose_udv(greens)
  svs_naive = F[2][end:-1:1]
  println("svs_udv, svs_naive, abs diff, rel diff")
  display(cat(2,svs_udv,svs_naive,absdiff(svs_naive, svs_udv), reldiff(svs_naive, svs_udv)))
end
