include("tests_gf_functions.jl")

"""
Boson action diff
"""
function test_calculate_boson_action_diff(p,l)
  old_hsfield = copy(p.hsfield)
  new_hsfield = copy(old_hsfield)

  site = rand(1:l.sites)
  slice = rand(1:p.slices)
  new_op = rand(3)
  # new_op = [100.,0.2,1234.56]
  new_hsfield[:,site,slice] = new_op[:]

  Sbef = calculate_boson_action(p,l,old_hsfield)
  Saft = calculate_boson_action(p,l,new_hsfield)
  dS_direct = Saft - Sbef
  dS = calculate_boson_action_diff(p,l,site,new_op,old_hsfield,slice)
  if dS==dS_direct
    error("Inconsistency between boson_action and boson_action_diff!")
  end
  return true
end
# test_calculate_boson_action_diff(p,l)
# Worked


"""
Green's function update after spatial lattice sweep
"""
function delta_naive(s::Stack, p::Params, l::Lattice, i::Int, new_op::Vector{Float64})
  V1 = interaction_matrix_exp(p,l,s.current_slice,-1.0)
  bkp = copy(p.hsfield[:,i,s.current_slice])
  p.hsfield[:,i,s.current_slice] = new_op[:]
  V2 = interaction_matrix_exp(p,l,s.current_slice)
  p.hsfield[:,i,s.current_slice] = bkp[:]
  return (V1 * V2 - eye(p.flv*l.sites,p.flv*l.sites))
end

function delta_i_naive(s::Stack, p::Params, l::Lattice, i::Int, new_op::Vector{Float64})
  return delta_naive(s,p,l,i,new_op)[i:l.sites:end,i:l.sites:end]
end

function test_delta_i_consistency(s,p,l)
  old_hsfield = copy(p.hsfield)

  site = rand(1:l.sites)
  slice = s.current_slice
  # new_op = [1.,2.,3.]
  new_op = rand(3)

  detratio = calculate_detratio(s,p,l,site,new_op)
  delta_n = delta_naive(s,p,l,site,new_op)
  delta_i_n = delta_i_naive(s,p,l,site,new_op)

  if !isapprox(delta_i_n,s.delta_i)
    error("Efficient calculation of delta_i inconsistent with naive calculation!")
  elseif delta_n[site:l.sites:end,site:l.sites:end] != s.delta_i
    error("Delta_i inconsistent with full Delta!")
  end
  return true
end
# Worked

function test_local_update_gf(s::Stack, p::Params, l::Lattice)
  p.hsfield = rand(3,l.sites,p.slices)
  s.current_slice = rand(1:p.slices)

  gfbefore = calculate_greens_udv_chkr(p,l,s.current_slice)
  s.greens = copy(gfbefore)

  for site in 1:l.sites
    new_op = rand(p.box, 3)
    detratio = calculate_detratio(s,p,l,site,new_op)
    update_greens!(s,p,l,site)
    p.hsfield[:,site,s.current_slice] = new_op[:]
  end

  gfafter = calculate_greens_udv_chkr(p,l,s.current_slice)

  println("Num exact vs updated")
  compare(s.greens, gfafter)
  # println("")
  # println("Before vs After (num exact)")
  # compare_greens(gfbefore, gfafter)
  nothing
end
# What about max rel dev (order 1)? Max abs dev around 1e-3 to 1e-2


function delta_naive_full_no_chkr(s::Stack, p::Params, l::Lattice, i::Int, new_op::Vector{Float64})
  V1 = interaction_matrix_exp(p,l,s.current_slice,-1.0)
  bkp = copy(p.hsfield[:,i,s.current_slice])
  p.hsfield[:,i,s.current_slice] = new_op[:]
  V2 = interaction_matrix_exp(p,l,s.current_slice)
  p.hsfield[:,i,s.current_slice] = bkp[:]

  return ( l.hopping_matrix_exp_inv * V1 * V2 * l.hopping_matrix_exp  - eye(p.flv*l.sites,p.flv*l.sites))
end

function delta_i_naive_full_no_chkr(s::Stack, p::Params, l::Lattice, i::Int, new_op::Vector{Float64})
  return delta_naive_full(s,p,l,i,new_op)[i:l.sites:end,i:l.sites:end]
end


function delta_naive_full(s::Stack, p::Params, l::Lattice, i::Int, new_op::Vector{Float64})
  Blinv = slice_matrix(p,l,s.current_slice,-1.)
  bkp = copy(p.hsfield[:,i,s.current_slice])
  p.hsfield[:,i,s.current_slice] = new_op[:]
  Blprime = slice_matrix(p,l,s.current_slice)
  p.hsfield[:,i,s.current_slice] = bkp[:]

  return ( Blinv * Blprime - eye(p.flv*l.sites,p.flv*l.sites))
end

function delta_i_naive_full(s::Stack, p::Params, l::Lattice, i::Int, new_op::Vector{Float64})
  return delta_naive_full(s,p,l,i,new_op)[i:l.sites:end,i:l.sites:end]
end

using PyPlot
function plot_delta_full_error(s,p,l)
  old_hsfield = copy(p.hsfield)

  site = rand(1:l.sites)
  slice = s.current_slice
  new_op = rand(3)

  dtrange = [0.1, 0.05, 0.01, 0.005, 0.001]
  maxabsdiff = zeros(length(dtrange))
  meanabsdiff = zeros(length(dtrange))
  maxeffreldiff = zeros(length(dtrange))
  meaneffreldiff = zeros(length(dtrange))

  maxabsdiff_i = zeros(length(dtrange))
  meanabsdiff_i = zeros(length(dtrange))
  maxeffreldiff_i = zeros(length(dtrange))
  meaneffreldiff_i = zeros(length(dtrange))
  for (k, dt) in enumerate(dtrange)
    p.delta_tau = dt

    delta_n = delta_naive(s,p,l,site,new_op)
    # delta_n_full = delta_naive_full_no_chkr(s,p,l,site,new_op)
    delta_n_full = delta_naive_full(s,p,l,site,new_op)

    delta_i_n = delta_i_naive(s,p,l,site,new_op)
    # delta_i_n_full = delta_i_naive_full_no_chkr(s,p,l,site,new_op)
    delta_i_n_full = delta_i_naive_full(s,p,l,site,new_op)

    maxabsdiff[k] = maximum(absdiff(delta_n, delta_n_full))
    meanabsdiff[k] = mean(absdiff(delta_n, delta_n_full))
    r = effreldiff(delta_n, delta_n_full)
    r[find(x->x==zero(x),delta_n)] = 0.
    maxeffreldiff[k] = maximum(r)
    meaneffreldiff[k] = mean(r)

    maxabsdiff_i[k] = maximum(absdiff(delta_i_n, delta_i_n_full))
    meanabsdiff_i[k] = mean(absdiff(delta_i_n, delta_i_n_full))
    r = effreldiff(delta_i_n, delta_i_n_full)
    r[find(x->x==zero(x),delta_i_n)] = 0.
    maxeffreldiff_i[k] = maximum(r)
    meaneffreldiff_i[k] = mean(r)

    # display(delta_n)
    # display(delta_n_full)
    # println("")
  end

  figure()
  loglog(dtrange, maxabsdiff, label="numerics (max error)")
  loglog(dtrange, meanabsdiff, "C9", label="numerics (mean error)")
  loglog(dtrange, dtrange, label="\$ O(\\Delta\\tau) \$")
  loglog(dtrange, dtrange.^2, label="\$ O(\\Delta\\tau^2) \$")
  title("Absolute error in \$ \\Delta \$ (L=$(l.L))")
  ylabel("error \$ (\\Delta, \\Delta_{full}) \$")
  xlabel("\$ \\Delta\\tau\$")
  legend()
  show()

  savefig("delta_abs_error_L_$(l.L).png")

  figure()
  loglog(dtrange, maxeffreldiff, label="numerics (max error)")
  loglog(dtrange, meaneffreldiff, "C9", label="numerics (mean error)")
  loglog(dtrange, dtrange, label="\$ O(\\Delta\\tau) \$")
  loglog(dtrange, dtrange.^2, label="\$ O(\\Delta\\tau^2) \$")
  title("Relative error in \$ \\Delta \$ (L=$(l.L))")
  ylabel("error \$ (\\Delta, \\Delta_{full}) \$")
  xlabel("\$ \\Delta\\tau\$")
  legend()
  show()

  savefig("delta_rel_error_L_$(l.L).png")

  figure()
  loglog(dtrange, maxabsdiff_i, label="numerics (max error)")
  loglog(dtrange, meanabsdiff_i, "C9", label="numerics (mean error)")
  loglog(dtrange, dtrange, label="\$ O(\\Delta\\tau) \$")
  loglog(dtrange, dtrange.^2, label="\$ O(\\Delta\\tau^2) \$")
  title("Absolute error in \$ \\Delta_i \$ (L=$(l.L))")
  ylabel("error \$ (\\Delta_i, \\Delta_{i,full}) \$")
  xlabel("\$ \\Delta\\tau\$")
  legend()
  show()

  savefig("delta_i_abs_error_L_$(l.L).png")

  figure()
  loglog(dtrange, maxeffreldiff_i, label="numerics (max error)")
  loglog(dtrange, meaneffreldiff_i, "C9", label="numerics (mean error)")
  loglog(dtrange, dtrange, label="\$ O(\\Delta\\tau) \$")
  loglog(dtrange, dtrange.^2, label="\$ O(\\Delta\\tau^2) \$")
  title("Relative error in \$ \\Delta_i \$ (L=$(l.L))")
  ylabel("error \$ (\\Delta_i, \\Delta_{i,full}) \$")
  xlabel("\$ \\Delta\\tau\$")
  legend()
  show()

  savefig("delta_i_rel_error_L_$(l.L).png")
end


function calculate_detratio_full(s::Stack, p::Params, l::Lattice, i::Int, new_op::Vector{Float64})
  s.delta_i = delta_i_naive_full(s,p,l,i,new_op)
  s.M = eye_flv + s.delta_i * (eye_flv - s.greens[i:l.sites:end,i:l.sites:end])
  return det(s.M)
end

function test_local_update_gf_full_delta(s::Stack, p::Params, l::Lattice)
  p.hsfield = rand(3,l.sites,p.slices)
  s.current_slice = rand(1:p.slices)

  gfbefore = calculate_greens_udv_chkr(p,l,s.current_slice)
  s.greens = copy(gfbefore)

  for site in 1:l.sites
    new_op = rand(p.box, 3)
    detratio = calculate_detratio_full(s,p,l,site,new_op)
    update_greens!(s,p,l,site)
    p.hsfield[:,site,s.current_slice] = new_op[:]
  end

  gfafter = calculate_greens_udv_chkr(p,l,s.current_slice)

  println("Num exact vs updated")
  compare(s.greens, gfafter)
  # println("")
  # println("Before vs After (num exact)")
  # compare_greens(gfbefore, gfafter)
  nothing
end

function update_greens_naive(s::Stack, p::Params, l::Lattice, i::Int, new_op::Vector{Float64})
  firstfactor = (eye_full - s.greens)
  secondfactor = (delta_naive_full(s,p,l,i,new_op) - s.greens)
  # return inv(eye_full + firstfactor * secondfactor) * s.greens
  return \(eye_full + firstfactor * secondfactor, s.greens)
end

function update_greens(s::Stack, p::Params, l::Lattice, i::Int)
  # first_term = (s.greens - eye_full)[:,i:l.sites:end] * inv(s.M)
  first_term = /((s.greens - eye_full)[:,i:l.sites:end], s.M)
  second_term = s.delta_i * s.greens[i:l.sites:end,:]
  return s.greens + first_term * second_term
end

function test_single_local_update_gf_naive(s::Stack, p::Params, l::Lattice)
  p.hsfield = rand(3,l.sites,p.slices)
  s.current_slice = rand(1:p.slices)
  site = rand(1:l.sites)

  gfbefore = calculate_greens_udv_chkr(p,l,s.current_slice)
  s.greens = copy(gfbefore)

  new_op = rand(p.box, 3)
  detratio = calculate_detratio(s,p,l,site,new_op)
  g_naive = update_greens_naive(s,p,l,site,new_op)
  update_greens!(s,p,l,site)
  p.hsfield[:,site,s.current_slice] = new_op[:]

  gfafter = calculate_greens_udv_chkr(p,l,s.current_slice)

  println("Num exact vs updated")
  compare(s.greens, gfafter)
  println("")
  println("Num exact vs direct update:")
  compare(s.greens, g_naive)
  # println("")
  # println("Before vs After (num exact)")
  # compare_greens(gfbefore, gfafter)
  nothing
end


function plot_gf_update_error(s,p,l)
  site = rand(1:l.sites)
  slice = rand(1:p.slices)
  new_op = rand(3)
  old_op = p.hsfield[:,site,slice]
  s.greens = calculate_greens_udv_chkr(p,l,slice)

  dtrange = [0.1, 0.05, 0.01, 0.005, 0.001]
  maxabsdiff = zeros(length(dtrange))
  meanabsdiff = zeros(length(dtrange))
  maxeffreldiff = zeros(length(dtrange))
  meaneffreldiff = zeros(length(dtrange))

  maxabsdiff_naive = zeros(length(dtrange))
  meanabsdiff_naive = zeros(length(dtrange))
  maxeffreldiff_naive = zeros(length(dtrange))
  meaneffreldiff_naive = zeros(length(dtrange))
  for (k, dt) in enumerate(dtrange)
    p.delta_tau = dt

    calculate_detratio(s,p,l,site,new_op)
    g_normal = update_greens(s,p,l,site)
    g_naive = update_greens_naive(s,p,l,site,new_op)
    p.hsfield[:,site,slice] = new_op[:]
    g = calculate_greens_udv_chkr(p,l,slice)
    p.hsfield[:,site,slice] = old_op[:]

    maxabsdiff[k] = maximum(absdiff(g_normal, g))
    meanabsdiff[k] = mean(absdiff(g_normal, g))
    r = effreldiff(g_normal, g)
    r[find(x->x==zero(x),g_normal)] = 0.
    maxeffreldiff[k] = maximum(r)
    meaneffreldiff[k] = mean(r)

    maxabsdiff_naive[k] = maximum(absdiff(g_naive, g))
    meanabsdiff_naive[k] = mean(absdiff(g_naive, g))
    r = effreldiff(g_naive, g)
    r[find(x->x==zero(x),g_naive)] = 0.
    maxeffreldiff_naive[k] = maximum(r)
    meaneffreldiff_naive[k] = mean(r)

    # display(delta_n)
    # display(delta_n_full)
    # println("")
  end

  figure()
  loglog(dtrange, maxabsdiff, label="numerics (max error)")
  loglog(dtrange, meanabsdiff, "C9", label="numerics (mean error)")
  loglog(dtrange, dtrange, label="\$ O(\\Delta\\tau) \$")
  loglog(dtrange, dtrange.^2, label="\$ O(\\Delta\\tau^2) \$")
  title("Absolute error in \$ G \$ after single local update (L=$(l.L))")
  ylabel("error \$ (G,G_{updated}) \$")
  xlabel("\$ \\Delta\\tau\$")
  legend()
  show()

  savefig("gupdated_normal_abs_error_L_$(l.L).png")

  figure()
  loglog(dtrange, maxeffreldiff, label="numerics (max error)")
  loglog(dtrange, meaneffreldiff, "C9", label="numerics (mean error)")
  loglog(dtrange, dtrange, label="\$ O(\\Delta\\tau) \$")
  loglog(dtrange, dtrange.^2, label="\$ O(\\Delta\\tau^2) \$")
  title("Relative error in \$ G \$ after single local update (L=$(l.L))")
  ylabel("error \$ (G,G_{updated}) \$")
  xlabel("\$ \\Delta\\tau\$")
  legend()
  show()

  savefig("gupdated_normal_rel_error_L_$(l.L).png")

  figure()
  loglog(dtrange, maxabsdiff_naive, label="numerics (max error)")
  loglog(dtrange, meanabsdiff_naive, "C9", label="numerics (mean error)")
  loglog(dtrange, dtrange, label="\$ O(\\Delta\\tau) \$")
  loglog(dtrange, dtrange.^2, label="\$ O(\\Delta\\tau^2) \$")
  title("Absolute error in \$ G \$ after single local update (L=$(l.L))")
  ylabel("error \$ (G,G_{naive,updated}) \$")
  xlabel("\$ \\Delta\\tau\$")
  legend()
  show()

  savefig("gupdated_naive_abs_error_L_$(l.L).png")

  figure()
  loglog(dtrange, maxeffreldiff_naive, label="numerics (max error)")
  loglog(dtrange, meaneffreldiff_naive, "C9", label="numerics (mean error)")
  loglog(dtrange, dtrange, label="\$ O(\\Delta\\tau) \$")
  loglog(dtrange, dtrange.^2, label="\$ O(\\Delta\\tau^2) \$")
  title("Relative error in \$ G \$ after single local update (L=$(l.L))")
  ylabel("error \$ (G,G_{naive,updated}) \$")
  xlabel("\$ \\Delta\\tau\$")
  legend()
  show()

  savefig("gupdated_naive_rel_error_L_$(l.L).png")
end