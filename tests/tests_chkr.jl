include("tests_gf_functions.jl")

using PyPlot
"""
Slice matrix
"""
function plot_slice_matrix_chkr_error_delta_tau_scaling(p::Parameters, l::Lattice)

  delta_tau_prev = p.delta_tau
  delta_tau_range = [.1, .01, .001, .0001]
  maxabsdiffs = zeros(length(delta_tau_range))
  maxreldiffs = zeros(length(delta_tau_range))
  for (k, delta_tau) in enumerate(delta_tau_range)
    p.delta_tau = delta_tau
    init_hopping_matrix_exp(p,l)
    init_checkerboard_matrices(p,l)

    B = slice_matrix_no_chkr(p,l,200)
    B_chkr = slice_matrix(p,l,200)

    maxabsdiffs[k] = maximum(absdiff(B,B_chkr))
    maxreldiffs[k] = maximum(reldiff(B,B_chkr))
  end

  # restore old delta_tau
  p.delta_tau = delta_tau_prev
  init_hopping_matrix_exp(p,l)
  init_checkerboard_matrices(p,l)

  figure()
  loglog(delta_tau_range,maxabsdiffs, label="numerics")
  loglog(delta_tau_range,delta_tau_range, label="\$ O(\\Delta\\tau) \$")
  loglog(delta_tau_range,delta_tau_range.^2, label="\$ O(\\Delta\\tau^2) \$")
  title("Checkerboard error in \$ B_n \$ (L=$(l.L))")
  ylabel("max abs diff \$ (B_n, B_n^{chkr}) \$")
  xlabel("\$ \\Delta\\tau\$")
  legend()

  savefig("chkr_slice_matrix_error_L_$(l.L).png")
end
# plot_slice_matrix_chkr_error_delta_tau_scaling(p,l)

function test_slice_matrix_chkr_speed(p::Parameters, l::Lattice, samples::Int=100)

  println("Sample size: ", samples)
  t = 0.
  t_chkr = 0.
  A = eye(Complex128, p.flv*l.sites)
  for k in 1:samples
    t += (@timed slice_matrix_no_chkr(p,l,200))[2]
    t_chkr += (@timed slice_matrix(p,l,200))[2]
  end

  @printf("avg time w/o checkerboard: %.1e\n", t/samples)
  @printf("avg time with checkerboard: %.1e\n", t_chkr/samples)
  @printf("difference: %.1e\n", (t_chkr-t)/samples)
  if t_chkr < t
    @printf("speedup factor: %.1f\n",  abs.(t/t_chkr))
  else
    @printf("slowdown(!) factor: %.1f\n",  abs.(t_chkr/t))
  end

end
# test_slice_matrix_chkr_speed(p,l,100)


"""
Green's function
"""
function plot_greens_chkr_error_delta_tau_scaling(p::Parameters, l::Lattice)

  delta_tau_prev = p.delta_tau
  delta_tau_range = [.1, .01, .001, .0001]
  maxabsdiffs = zeros(length(delta_tau_range))
  maxreldiffs = zeros(length(delta_tau_range))
  for (k, delta_tau) in enumerate(delta_tau_range)
    p.delta_tau = delta_tau
    init_hopping_matrix_exp(p,l)
    init_checkerboard_matrices(p,l)

    greens = calculate_greens_and_logdet(p,l,1)[1]
    greens_chkr = calculate_greens_and_logdet_chkr(p,l,1)[1]

    maxabsdiffs[k] = maximum(absdiff(greens,greens_chkr))
    maxreldiffs[k] = maximum(reldiff(greens,greens_chkr))
  end

  # restore old delta_tau
  p.delta_tau = delta_tau_prev
  init_hopping_matrix_exp(p,l)
  init_checkerboard_matrices(p,l)

  figure()
  loglog(delta_tau_range,maxabsdiffs, label="numerics")
  loglog(delta_tau_range,delta_tau_range, label="\$ O(\\Delta\\tau) \$")
  loglog(delta_tau_range,delta_tau_range.^2, label="\$ O(\\Delta\\tau^2) \$")
  title("Checkerboard error in \$ G \$ (L=$(l.L))")
  ylabel("max abs diff \$ (G, G_{chkr}) \$")
  xlabel("\$ \\Delta\\tau\$")
  legend()

  savefig("chkr_greens_error_L_$(l.L).png")

end


function test_greens_chkr_speed(p::Parameters, l::Lattice, samples::Int=10)

  println("Sample size: ", samples)
  t = 0.
  t_chkr = 0.
  A = eye(Complex128, p.flv*l.sites)
  for k in 1:samples
    t += (@timed calculate_greens_udv(p,l,1))[2]
    t_chkr += (@timed calculate_greens_udv_chkr(p,l,1))[2]
  end

  @printf("avg time w/o checkerboard: %.1e\n", t/samples)
  @printf("avg time with checkerboard: %.1e\n", t_chkr/samples)
  @printf("difference: %.1e\n", (t_chkr-t)/samples)
  if t_chkr < t
    @printf("speedup factor: %.1f\n",  abs.(t/t_chkr))
  else
    @printf("slowdown(!) factor: %.1f\n",  abs.(t_chkr/t))
  end

end


"""
Time slice wrapping
"""
function wrap_greens_chkr(p::Parameters, l::Lattice, gf::Array{Complex{Float64},2},slice::Int,direction::Int,n::Int)
  current_slice = slice
  g = gf
  assert(abs.(direction) == 1)

  for k in 1:n
    g = wrap_greens_chkr(p,l,g, current_slice, direction)
    current_slice += direction
  end

  return g
end

function test_gf_wrapping_chkr(p::Parameters, l::Lattice)
  slice = rand(10:p.slices)
  gf = calculate_greens_udv_chkr(p,l,slice)

  # wrapping down
  println("")
  println("Downwards\n\n")
  gfwrapped = wrap_greens_chkr(p,l,gf, slice, -1)
  gfwrapped2 = wrap_greens_chkr(p,l,gf, slice, -1, 2)
  gfwrapped3 = wrap_greens_chkr(p,l,gf, slice, -1, 3)
  gfwrapped10 = wrap_greens_chkr(p,l,gf, slice, -1, 10)
  gfexact = calculate_greens_udv_chkr(p,l,slice - 1)
  gfexact2 = calculate_greens_udv_chkr(p,l,slice - 2)
  gfexact3 = calculate_greens_udv_chkr(p,l,slice - 3)
  gfexact10 = calculate_greens_udv_chkr(p,l,slice - 10)

  println("Comparing wrapped down vs num exact")
  compare(gfwrapped,gfexact)
  println("")
  println("Comparing twice wrapped down vs num exact")
  compare(gfwrapped2,gfexact2)
  println("")
  println("Comparing 3-times wrapped down vs num exact")
  compare(gfwrapped3,gfexact3)
  println("")
  println("Comparing 10-times wrapped up vs num exact")
  compare(gfwrapped10,gfexact10)

  # wrapping up
  println("")
  println("")
  println("Upwards\n\n")
  gfwrapped = wrap_greens_chkr(p,l,gf, slice, 1)
  gfwrapped2 = wrap_greens_chkr(p,l,gf,slice, 1, 2)
  gfwrapped3 = wrap_greens_chkr(p,l,gf,slice, 1, 3)
  gfwrapped10 = wrap_greens_chkr(p,l,gf,slice, 1, 10)
  gfexact = calculate_greens_udv_chkr(p,l,slice + 1)
  gfexact2 = calculate_greens_udv_chkr(p,l,slice + 2)
  gfexact3 = calculate_greens_udv_chkr(p,l,slice + 3)
  gfexact10 = calculate_greens_udv_chkr(p,l,slice + 10)

  println("")
  println("Comparing wrapped up vs num exact")
  compare(gfwrapped,gfexact)
  println("")
  println("Comparing twice wrapped up vs num exact")
  compare(gfwrapped2,gfexact2)
  println("")
  println("Comparing 3-times wrapped up vs num exact")
  compare(gfwrapped3,gfexact3)
  println("")
  println("Comparing 10-times wrapped up vs num exact")
  compare(gfwrapped10,gfexact10)

  nothing
end
# worked



"""
multiply right left (inverse) check
"""
function test_multiply_slice_matrix(s,p,l)
  id = eye_full
  println("B^-1 * B * 1")
  if !compare(multiply_slice_matrix_inv_left(p,l,1,multiply_slice_matrix_left(p,l,1,id)),id)
    warn("B^-1 * B * 1 != 1")
  end
  println("B * B^-1 * 1")
  if !compare(multiply_slice_matrix_left(p,l,1,multiply_slice_matrix_inv_left(p,l,1,id)),id)
    warn("B * B^-1 * 1 != 1")
  end
  println("1 * B * B^-1")
  if !compare(multiply_slice_matrix_inv_right(p,l,1,multiply_slice_matrix_right(p,l,1,id)),id)
    warn("1 * B * B^-1 != 1")
  end
  println("1 * B^-1 * B")
  if !compare(multiply_slice_matrix_right(p,l,1,multiply_slice_matrix_inv_right(p,l,1,id)),id)
    warn("1 * B^-1 * B != 1")
  end

  println("")
  println("Mixed")
  println("")

  println("(B * 1) * B^-1")
  if !compare(multiply_slice_matrix_inv_right(p,l,1,multiply_slice_matrix_left(p,l,1,id)),id)
    warn("(B * 1) * B^-1 != 1")
  end
  println("B^-1 * (1 * B)")
  if !compare(multiply_slice_matrix_inv_left(p,l,1,multiply_slice_matrix_right(p,l,1,id)),id)
  warn("B * (1 * B^-1) != 1")
  end
    println("(B^-1 * 1) * B")
  if !compare(multiply_slice_matrix_right(p,l,1,multiply_slice_matrix_inv_left(p,l,1,id)),id)
    warn("(B^-1 * 1) * B != 1")
  end
  println("B * (1 * B^-1)")
  if !compare(multiply_slice_matrix_left(p,l,1,multiply_slice_matrix_inv_right(p,l,1,id)),id)
    warn("B * (1 * B^-1) != 1")
  end
end