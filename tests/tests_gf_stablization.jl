include("tests_gf_functions.jl")

"""
Singular values stabilization
"""
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
plot_lowest_sv_of_slice_matrix_chain_vs_safe_mult(p,l)
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
