include("tests_gf_functions.jl")

"""
Inversion in greens calculation
"""
function test_greens_udv_vs_naive_vs_naiveudvinv(p::Parameters, l::Lattice)
  # slice = rand(1:p.slices)
  slice = 15
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



using PyPlot
"""
Stack logic consistency
"""
function plot_gf_error_propagation(s::Stack, p::Parameters, l::Lattice)
  initialize_stack(s,p,l)
  build_stack(s,p,l)
  propagate(s,p,l)

  mean_absdiff = zeros(2*p.slices)
  mean_reldiff = zeros(2*p.slices)
  max_absdiff = zeros(2*p.slices)
  max_reldiff = zeros(2*p.slices)
  slices = Vector{Int}(2*p.slices)
  for n in 1:2*p.slices
    propagate(s,p,l)
    println("")
    println("current slice: ", s.current_slice, "(", n,")")

    g = calculate_greens_and_logdet_chkr(p,l,s.current_slice)[1]
    mean_absdiff[n] = mean(absdiff(g,s.greens))
    println("mean absdiff: ", mean_absdiff[n])
    mean_reldiff[n] = mean(effreldiff(g,s.greens))
    println("mean reldiff: ", mean_reldiff[n])
    max_absdiff[n] = maximum(absdiff(g,s.greens))
    println("max absdiff: ", max_absdiff[n])
    max_reldiff[n] = maximum(effreldiff(g,s.greens))
    println("max reldiff: ", max_reldiff[n])
    slices[n] = s.current_slice
  end

  write("mean_absdiff_L_$(l.L)_safe_mult_$(p.safe_mult).bin", mean_absdiff)
  write("mean_effreldiff_L_$(l.L)_safe_mult_$(p.safe_mult).bin", mean_reldiff)
  write("max_absdiff_L_$(l.L)_safe_mult_$(p.safe_mult).bin", max_absdiff)
  write("max_effreldiff_L_$(l.L)_safe_mult_$(p.safe_mult).bin", max_reldiff)

  # mean devs
  fig = figure(figsize=(20,10))
  ax1 = fig[:add_subplot](2,1,1)
  ax1[:plot](mean_absdiff,"C0")
  ax2 = fig[:add_subplot](2,1,2)
  ax2[:plot](mean_reldiff,"C3")
  subplots_adjust(wspace = 0.0)

  ax1[:set_ylabel]("mean absdiff")
  ax2[:set_ylabel]("mean effreldiff")
  fig[:text](.5, .95, "Mean errors (L=$(l.L), safe_mult=$(p.safe_mult))", ha="center")
  # fig[:text](.5, .03, "time slice", ha="center")

  # max devs
  fig2 = figure(figsize=(20,10))
  ax1 = fig2[:add_subplot](2,1,1)
  ax1[:plot](max_absdiff,"C0")
  ax2 = fig2[:add_subplot](2,1,2)
  ax2[:plot](max_reldiff,"C3")
  subplots_adjust(wspace = 0.0)

  ax1[:set_ylabel]("max absdiff")
  ax2[:set_ylabel]("max effreldiff")
  fig2[:text](.5, .95, "Maximum errors (L=$(l.L), safe_mult=$(p.safe_mult))", ha="center")
  fig2[:text](.5, .03, "time slice", ha="center")

  fig[:savefig]("greens_error_mean L_$(l.L)_safe_mult_$(p.safe_mult).png")
  fig2[:savefig]("greens_error_max L_$(l.L)_safe_mult_$(p.safe_mult).png")
  # show()
end
# plot_gf_error_propagation(s,p,l)


"""
Green's function singular values (full determinant calculation)
"""
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


function test_greens_det(s::Stack, p::Parameters, l::Lattice)
  greens, ldet = calculate_greens_and_logdet_chkr(p,l,s.current_slice)
  # greens, ldet = measure_greens_and_logdet(p,l)
  greens_udv, ldet_udv = calculate_greens_and_logdet_chkr_udv(p,l,s.current_slice)
  ldet_naive = logdet(greens)
  println("logdet (QR): ", ldet)
  println("logdet SVD: ", ldet_udv)
  println("logdet naive: ", ldet_naive)
  println("")
  println("det (QR): ", exp(ldet))
  println("det SVD: ", exp(ldet_udv))
  println("det naive: ", exp(ldet_naive))
end


function test_calculate_greens_symmetry(s::Stack, p::Parameters, l::Lattice)
  U, D, Vt = decompose_udv!(rand(Complex128, p.flv*l.sites, p.flv*l.sites))

  # U,D,Vt right = eye
  s.Ul = copy(U)
  s.Dl = copy(D)
  s.Vtl = copy(Vt)
  s.Ur = eye(Complex128, p.flv*l.sites, p.flv*l.sites)
  s.Dr = ones(p.flv*l.sites)
  s.Vtr = eye(Complex128, p.flv*l.sites, p.flv*l.sites)
  gl = calculate_greens(s,p,l)

  # U,D,Vt left = eye
  s.Ur = copy(U)
  s.Dr = copy(D)
  s.Vtr = copy(Vt)
  s.Ul = eye(Complex128, p.flv*l.sites, p.flv*l.sites)
  s.Dl = ones(p.flv*l.sites)
  s.Vtl = eye(Complex128, p.flv*l.sites, p.flv*l.sites)
  gr = calculate_greens(s,p,l)

  compare(gl,gr)

end
# Worked
