using PyPlot
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
  title("Checkerboard error in \$ B_n \$")
  ylabel("max abs diff \$ (B_n, B_n^{chkr}) \$")
  xlabel("\$ \\Delta\\tau\$")
  legend()

end
plot_slice_matrix_chkr_error_delta_tau_scaling(p,l)
# absdiff seems to scale with p.delta_tau, shouldnt it be more like p.delta_tau^2?
# Update: After forgotten l.chkr_mu has been added, chkr error in B_n is effectively zero (too small?!). Why?


function test_slice_matrix_chkr_speed(p::Parameters, l::Lattice, samples::Int=100)

  println("Sample size: ", samples)
  t = 0.
  t_chkr = 0.
  for k in 1:samples
    t += (@timed slice_matrix_no_chkr(p,l,200))[2]
    t_chkr += (@timed slice_matrix(p,l,200))[2]
  end

  println("avg time w/o checkerboard: ", t/samples)
  println("avg time with checkerboard: ", t_chkr/samples)
  println("diff relto no chkr: ", (t_chkr-t)/samples)
  if t_chkr < t
    @printf("rel relto no chkr: -%.1f%%",  abs(t_chkr-t)/t * 100)
  else
    @printf("rel relto no chkr: +%.1f%%",  abs(t_chkr-t)/t * 100)
  end

end
test_slice_matrix_chkr_speed(p,l,500)
# TODO: CHECKERBOARD NOT FASTER THAN NO-CHECKERBOARD! Must be fixed!
