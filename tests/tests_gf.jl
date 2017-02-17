include("tests_gf_functions.jl")

"""
Inversion in greens calculation
"""
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



using PyPlot
"""
Stack logic consistency
"""
function plot_gf_error_propagation(s::Stack, p::Parameters, l::Lattice)
  # p.slices = 200
  # p.safe_mult = 1
  p.hsfield = rand(3,l.sites,p.slices)
  initialize_stack(s,p,l)
  build_stack(s,p,l)
  mean_dev = Vector{Float64}(p.slices)
  mean_rel_dev = Vector{Float64}(p.slices)
  max_dev = Vector{Float64}(p.slices)
  max_rel_dev = Vector{Float64}(p.slices)
  slices = Vector{Int}(p.slices)
  println("Starting down sweep")
  for n in 1:p.slices
    println(n)
    propagate(s,p,l)
    gf = calculate_greens_udv(p,l,s.current_slice)
    mean_dev[n] = mean(absdiff(gf,s.greens))
    mean_rel_dev[n] = mean(reldiff(gf,s.greens))
    max_dev[n] = maximum(absdiff(gf,s.greens))
    max_rel_dev[n] = maximum(reldiff(gf,s.greens))
    slices[n] = s.current_slice
  end

  mean_devUP = Vector{Float64}(p.slices)
  mean_rel_devUP = Vector{Float64}(p.slices)
  max_devUP = Vector{Float64}(p.slices)
  max_rel_devUP = Vector{Float64}(p.slices)
  slicesUP = Vector{Int}(p.slices)
  println("Starting up sweep")
  for n in 1:p.slices
    println(n)
    propagate(s,p,l)
    gf = calculate_greens_udv(p,l,s.current_slice)
    mean_devUP[n] = maximum(absdiff(gf,s.greens))
    mean_rel_devUP[n] = maximum(reldiff(gf,s.greens))
    max_devUP[n] = maximum(absdiff(gf,s.greens))
    max_rel_devUP[n] = maximum(reldiff(gf,s.greens))
    slicesUP[n] = s.current_slice
  end

  # mean devs
  fig, (ax1, ax2, ax3, ax4) = subplots(2,2,sharey=true,figsize=(20,10))
  ax1[:plot](slices,mean_dev,"C0")
  ax1[:invert_xaxis]()
  ax3[:plot](slicesUP,mean_devUP,"C0")
  ax2[:plot](slices,mean_rel_dev,"C3")
  ax2[:invert_xaxis]()
  ax4[:plot](slicesUP,mean_rel_devUP,"C3")
  subplots_adjust(wspace = 0.0)

  ax1[:set_ylabel]("mean absdiff")
  ax2[:set_ylabel]("mean reldiff")
  fig[:text](.5, .95, "Mean errors (L=$(l.L), safe_mult=$(p.safe_mult))", ha="center")
  fig[:text](.5, .03, "time slice", ha="center")

  # max devs
  fig, (ax1, ax2, ax3, ax4) = subplots(2,2,sharey=true,figsize=(20,10))
  ax1[:plot](slices,max_dev,"C0")
  ax1[:invert_xaxis]()
  ax3[:plot](slicesUP,max_devUP,"C0")
  ax2[:plot](slices,max_rel_dev,"C3")
  ax2[:invert_xaxis]()
  ax4[:plot](slicesUP,max_rel_devUP,"C3")
  subplots_adjust(wspace = 0.0)

  ax1[:set_ylabel]("max absdiff")
  ax2[:set_ylabel]("max reldiff")
  fig[:text](.5, .95, "Maximum errors (L=$(l.L), safe_mult=$(p.safe_mult))", ha="center")
  fig[:text](.5, .03, "time slice", ha="center")

  show()
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
