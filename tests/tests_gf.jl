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
function plot_gf_error_propagation(s::Stack, p::Parameters, l::Lattice, mode::String="chkr")
  # p.slices = 200
  # p.safe_mult = 1
  p.hsfield = rand(3,l.sites,p.slices)
  initialize_stack(s,p,l)
  build_stack(s,p,l)
  gf = similar(s.greens)

  mean_absdiff = Vector{Float64}(p.slices)
  mean_reldiff = Vector{Float64}(p.slices)
  max_absdiff = Vector{Float64}(p.slices)
  max_reldiff = Vector{Float64}(p.slices)
  slices = Vector{Int}(p.slices)
  println("Starting down sweep")
  for n in 1:p.slices
    println(n)
    propagate(s,p,l)

    if mode == "chkr"
      gf = calculate_greens_udv_chkr(p,l,s.current_slice)
    else
      gf = calculate_greens_udv(p,l,s.current_slice)
    end

    mean_absdiff[n] = mean(absdiff(gf,s.greens))
    mean_reldiff[n] = mean(reldiff(gf,s.greens))
    max_absdiff[n] = maximum(absdiff(gf,s.greens))
    max_reldiff[n] = maximum(reldiff(gf,s.greens))
    slices[n] = s.current_slice
  end

  mean_absdiffUP = Vector{Float64}(p.slices)
  mean_reldiffUP = Vector{Float64}(p.slices)
  max_absdiffUP = Vector{Float64}(p.slices)
  max_reldiffUP = Vector{Float64}(p.slices)
  slicesUP = Vector{Int}(p.slices)
  println("Starting up sweep")
  for n in 1:p.slices
    println(n)
    propagate(s,p,l)

    if mode == "chkr"
      gf = calculate_greens_udv_chkr(p,l,s.current_slice)
    else
      gf = calculate_greens_udv(p,l,s.current_slice)
    end

    mean_absdiffUP[n] = maximum(absdiff(gf,s.greens))
    mean_reldiffUP[n] = maximum(reldiff(gf,s.greens))
    max_absdiffUP[n] = maximum(absdiff(gf,s.greens))
    max_reldiffUP[n] = maximum(reldiff(gf,s.greens))
    slicesUP[n] = s.current_slice
  end

  if mode == "chkr"
    write("mean_absdiff_chkr_L_$(l.L)_safe_mult_$(p.safe_mult).bin", [mean_absdiff; mean_absdiffUP])
    write("mean_reldiff_chkr_L_$(l.L)_safe_mult_$(p.safe_mult).bin", [mean_reldiff; mean_reldiffUP])
    write("max_absdiff_chkr_L_$(l.L)_safe_mult_$(p.safe_mult).bin", [max_absdiff; max_absdiffUP])
    write("max_reldiff_chkr_L_$(l.L)_safe_mult_$(p.safe_mult).bin", [max_reldiff; max_reldiffUP])
  else
    write("mean_absdiff_L_$(l.L)_safe_mult_$(p.safe_mult).bin", [mean_absdiff; mean_absdiffUP])
    write("mean_reldiff_L_$(l.L)_safe_mult_$(p.safe_mult).bin", [mean_reldiff; mean_reldiffUP])
    write("max_absdiff_L_$(l.L)_safe_mult_$(p.safe_mult).bin", [max_absdiff; max_absdiffUP])
    write("max_reldiff_L_$(l.L)_safe_mult_$(p.safe_mult).bin", [max_reldiff; max_reldiffUP])
  end

  # mean devs
  fig = figure(figsize=(20,10))
  ax1 = fig[:add_subplot](2,2,1)
  ax1[:plot](slices,mean_absdiff,"C0")
  ax1[:invert_xaxis]()
  ax3 = fig[:add_subplot](2,2,2,sharey=ax1)
  ax3[:plot](slicesUP,mean_absdiffUP,"C0")
  setp(ax3[:get_yticklabels](), visible=false)
  ax2 = fig[:add_subplot](2,2,3)
  ax2[:plot](slices,mean_reldiff,"C3")
  ax2[:invert_xaxis]()
  ax4 = fig[:add_subplot](2,2,4,sharey=ax2)
  ax4[:plot](slicesUP,mean_reldiffUP,"C3")
  setp(ax4[:get_yticklabels](), visible=false)
  subplots_adjust(wspace = 0.0)

  ax1[:set_ylabel]("mean absdiff")
  ax2[:set_ylabel]("mean reldiff")
  fig[:text](.5, .95, "Mean errors" * (mode == "chkr" ? " chkr " : " ") * "(L=$(l.L), safe_mult=$(p.safe_mult))", ha="center")
  fig[:text](.5, .03, "time slice", ha="center")

  # max devs
  fig2 = figure(figsize=(20,10))
  ax1 = fig2[:add_subplot](2,2,1)
  ax1[:plot](slices,max_absdiff,"C0")
  ax1[:invert_xaxis]()
  ax3 = fig2[:add_subplot](2,2,2,sharey=ax1)
  ax3[:plot](slicesUP,max_absdiffUP,"C0")
  setp(ax3[:get_yticklabels](), visible=false)
  ax2 = fig2[:add_subplot](2,2,3)
  ax2[:plot](slices,max_reldiff,"C3")
  ax2[:invert_xaxis]()
  ax4 = fig2[:add_subplot](2,2,4,sharey=ax2)
  ax4[:plot](slicesUP,max_reldiffUP,"C3")
  setp(ax4[:get_yticklabels](), visible=false)
  subplots_adjust(wspace = 0.0)

  ax1[:set_ylabel]("max absdiff")
  ax2[:set_ylabel]("max reldiff")
  fig2[:text](.5, .95, "Maximum errors" * (mode == "chkr" ? " chkr " : " ") * "(L=$(l.L), safe_mult=$(p.safe_mult))", ha="center")
  fig2[:text](.5, .03, "time slice", ha="center")

  fig[:savefig]("greens_error_mean" * (mode == "chkr" ? "_chkr_" : "_" ) * "L_$(l.L)_safe_mult_$(p.safe_mult).png")
  fig2[:savefig]("greens_error_max" * (mode == "chkr" ? "_chkr_" : "_" ) * "L_$(l.L)_safe_mult_$(p.safe_mult).png")
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


function test_greens_det_naive(s::Stack, p::Parameters, l::Lattice)
  greens, det_udv, svs_udv = calculate_greens_and_det_and_svs_udv(p,l,s.current_slice)
  det_naive = det(greens)
  F = decompose_udv(greens)
  svs_naive = F[2][end:-1:1]
  println("svs_udv, svs_naive, abs diff, rel diff")
  display(cat(2,svs_udv,svs_naive,absdiff(svs_naive, svs_udv), reldiff(svs_naive, svs_udv)))
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
