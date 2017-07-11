include("tests_gf_functions.jl")


using PyPlot
using PyCall
@pyimport matplotlib.ticker as ticker
"""
Singular values stabilization
"""
function plot_svs_of_slice_matrix_chain(p::Parameters, l::Lattice)
  T = calculate_slice_matrix_chain_naive(p,l,1,p.slices)
  svs = T[2]
  fig = figure(figsize=(20,7))
  b = fig[:add_subplot](131)
  b[:yaxis][:set_major_locator](ticker.MaxNLocator(symmetric=true))
  b[:plot](svs[:,:]')
  b[:set_xlabel]("Inverse temperature \$\\beta\$")
  b[:set_ylabel]("log singular values of \$B(\\beta,0)\$")
  b[:set_title]("Naive")

  T = calculate_slice_matrix_chain_udv(p,l,1,p.slices,1)
  svs = T[4]
  a = fig[:add_subplot](132, sharey=b, sharex=b)
  a[:plot](svs[:,:]')
  setp(a[:get_yticklabels](), visible=false)
  a[:set_xlabel]("Inverse temperature \$\\beta\$")
  a[:set_title]("Stabilize, no chkr")

  T = calculate_slice_matrix_chain_udv_chkr(p,l,1,p.slices,1)
  svs = T[4]
  c = fig[:add_subplot](133, sharey=b, sharex=b)
  c[:plot](svs[:,:]')
  setp(c[:get_yticklabels](), visible=false)
  c[:set_xlabel]("Inverse temperature \$\\beta\$")
  c[:set_title]("Stabilize, with chkr")

  # tight_layout()
  subplots_adjust(wspace = 0.0)
  show()
  println(maximum(svs))
  nothing
end
# plot_svs_of_slice_matrix_chain(p,l)
# Worked


"""
Safe_mult
"""
# Direct "safe_mult" test
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
# plot_lowest_sv_of_slice_matrix_chain_vs_safe_mult(p,l)
# Worked.


# p.safe_mult test (as part of stack logic)
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



"""
Time slice wrapping
"""
function wrap_greens(p::Parameters, l::Lattice, gf::Array{Complex{Float64},2},slice::Int,direction::Int)
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

function wrap_greens2(p::Parameters, l::Lattice, gf::Array{Complex{Float64},2},slice::Int,direction::Int)
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
  gfwrapped1 = wrap_greens(p,l,gf,slice,-1)
  gfwrapped2 = wrap_greens2(p,l,gf,slice,-1)

  println("Comparing wrapped down vs wrapped down2")
  compare_greens(gfwrapped1,gfwrapped2)
end
# Worked! Found mistake in slice_matrix_no_chkr in stack.jl


function test_gf_wrapping(p::Parameters, l::Lattice)
  slice = rand(3:p.slices-2)
  println("Slice: ", slice)
  # slice = 3
  gf = calculate_greens_udv(p,l,slice)

  # wrapping down
  gfwrapped = wrap_greens(p,l,gf,slice,-1)
  gfwrapped2 = wrap_greens(p,l,gfwrapped,slice-1,-1)
  gfexact = calculate_greens_udv(p,l,slice - 1)
  gfexact2 = calculate_greens_udv(p,l,slice - 2)

  println("Comparing wrapped down vs num exact")
  compare_greens(gfwrapped,gfexact)
  println("")
  println("Comparing twice wrapped down vs num exact")
  compare_greens(gfwrapped2,gfexact2)

  # wrapping up
  gfwrapped = wrap_greens(p,l,gf,slice,1)
  gfwrapped2 = wrap_greens(p,l,gfwrapped,slice+1,1)
  gfexact = calculate_greens_udv(p,l,slice + 1)
  gfexact2 = calculate_greens_udv(p,l,slice + 2)

  println("")
  println("")
  println("Comparing wrapped up vs num exact")
  compare_greens(gfwrapped,gfexact)
  println("")
  println("Comparing twice wrapped up vs num exact")
  compare_greens(gfwrapped2,gfexact2)

  nothing
end

function test_gf_wrapping_slice_dependency(p::Parameters, l::Lattice)
  for slice in [1; 2; 3; convert(Array{Int},floor(linspace(4, p.slices-4, 10))); p.slices-3; p.slices-2; p.slices-1 ]
    println("Slice: ", slice)
    # slice = 3
    gf = calculate_greens_udv(p,l,slice)

    # wrapping up
    gfwrapped = wrap_greens(p,l,gf,slice,1)
    gfexact = calculate_greens_udv(p,l,slice + 1)

    println("Comparing wrapped up vs num exact")
    compare_greens(gfwrapped,gfexact)
    println("")
    println("")

  end

  nothing
end

using PyPlot
function plot_gf_wrapping_slice_dependency(p::Parameters, l::Lattice)
  gmaxabs = Float64[]
  gmaxrel = Float64[]
  slices = [1; 2; 3; convert(Array{Int},floor(linspace(4, p.slices-4, 5))); p.slices-3; p.slices-2; p.slices-1 ]
  # slices = [1, 2, 3]
  for slice in slices
    println("Slice: ", slice)
    # slice = 3
    gf = calculate_greens_udv(p,l,slice)

    # wrapping up
    gfwrapped = wrap_greens(p,l,gf,slice,1)
    gfexact = calculate_greens_udv(p,l,slice + 1)

    push!(gmaxabs, maximum(absdiff(gfwrapped,gfexact)))
    push!(gmaxrel, maximum(effreldiff(gfwrapped,gfexact)))
    println("maxabs: ", gmaxabs[end])
    println("maxrel: ", gmaxrel[end])
    println("")

  end

  fig, ax = subplots(1, 2, figsize=(10,4))
  ax[1][:plot](slices, gmaxabs, ".-")
  ax[1][:set_xlabel]("start slice")
  ax[1][:set_ylabel]("upwards wrapping error (absolute)")
  ax[1][:set_ylim]([-1e-4, 1e-3])

  ax[2][:plot](slices, gmaxrel, ".-", color="C1")
  ax[2][:set_xlabel]("start slice")
  ax[2][:set_ylabel]("upwards wrapping error (relative)")
  ax[2][:set_ylim]([-1e-4, 1e-3])
  tight_layout()

  nothing
end

function plot_gf_maxmeanabs_slice_dependency(p::Parameters, l::Lattice)
  gmaxabs = Float64[]
  gmeanabs = Float64[]
  slices = [1; 2; 3; convert(Array{Int},floor(linspace(4, p.slices-4, 10))); p.slices-3; p.slices-2; p.slices-1 ]
  # slices = [1, 2, 3]
  for slice in slices
    println("Slice: ", slice)
    # slice = 3
    gf = calculate_greens_udv(p,l,slice)

    push!(gmaxabs, maximum(abs(gf)))
    push!(gmeanabs, mean(abs(gf)))
    println("maxabs: ", gmaxabs[end])
    println("meanabs: ", gmeanabs[end])
    println("")

  end

  fig, ax = subplots(1, 2, figsize=(10,4))
  ax[1][:plot](slices, gmaxabs, ".-")
  ax[1][:set_xlabel]("start slice")
  ax[1][:set_ylabel]("maxabs entry of GF")
  ax[1][:set_ylim]([-1e-4, 1.0])

  ax[2][:plot](slices, gmeanabs, ".-", color="C1")
  ax[2][:set_xlabel]("start slice")
  ax[2][:set_ylabel]("meanabs entry of GF")
  ax[2][:set_ylim]([-1e-4, 1.0])
  tight_layout()

  nothing
end

function test_gf_wrapping_chkr(p::Parameters, l::Lattice)
  slice = rand(3:p.slices-2)
  # slice = 10
  gf = calculate_greens_udv_chkr(p,l,slice)

  # wrapping down
  gfwrapped = wrap_greens_chkr(gf,slice,-1)
  gfwrapped2 = wrap_greens_chkr(gfwrapped,slice-1,-1)
  gfexact = calculate_greens_udv_chkr(p,l,slice - 1)
  gfexact2 = calculate_greens_udv_chkr(p,l,slice - 2)

  println("Comparing wrapped down vs num exact")
  compare(gfwrapped,gfexact)
  println("")
  println("Comparing twice wrapped down vs num exact")
  compare(gfwrapped2,gfexact2)

  # wrapping up
  gfwrapped = wrap_greens_chkr(gf,slice,1)
  gfwrapped2 = wrap_greens_chkr(gfwrapped,slice+1,1)
  gfexact = calculate_greens_udv_chkr(p,l,slice + 1)
  gfexact2 = calculate_greens_udv_chkr(p,l,slice + 2)

  println("")
  println("")
  println("Comparing wrapped up vs num exact")
  compare(gfwrapped,gfexact)
  println("")
  println("Comparing twice wrapped up vs num exact")
  compare(gfwrapped2,gfexact2)

  nothing
end

function test_gf_wrapping_chkr_naive(p::Parameters, l::Lattice)
  slice = rand(3:p.slices-2)
  # slice = 10
  gf = calculate_greens_udv_chkr(p,l,slice)

  # wrapping down
  gfwrapped = wrap_greens_chkr_naive(gf,slice,-1)
  gfwrapped2 = wrap_greens_chkr_naive(gfwrapped,slice-1,-1)
  gfexact = calculate_greens_udv_chkr(p,l,slice - 1)
  gfexact2 = calculate_greens_udv_chkr(p,l,slice - 2)

  println("Comparing wrapped down vs num exact")
  compare(gfwrapped,gfexact)
  println("")
  println("Comparing twice wrapped down vs num exact")
  compare(gfwrapped2,gfexact2)

  # wrapping up
  gfwrapped = wrap_greens_chkr_naive(gf,slice,1)
  gfwrapped2 = wrap_greens_chkr_naive(gfwrapped,slice+1,1)
  gfexact = calculate_greens_udv_chkr(p,l,slice + 1)
  gfexact2 = calculate_greens_udv_chkr(p,l,slice + 2)

  println("")
  println("")
  println("Comparing wrapped up vs num exact")
  compare(gfwrapped,gfexact)
  println("")
  println("Comparing twice wrapped up vs num exact")
  compare(gfwrapped2,gfexact2)

  nothing
end