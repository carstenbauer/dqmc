function test_slice_matrix_chain_svd_vs_qr(p::Parameters, l::Lattice)
  start = rand(1:Int(floor(p.slices/2)))
  stop = rand(start+1:p.slices)
  println("B(",stop,",",start,")")

  Bcqr = calculate_slice_matrix_chain_qr(p,l,1,p.slices);
  Bcsvd = calculate_slice_matrix_chain_udv(p,l,1,p.slices);
  Bsvd = Bcsvd[1] * diagm(Bcsvd[2]) * Bcsvd[3];
  Bqr = Bcqr[1] * diagm(Bcqr[2]) * Bcqr[3];

  println("Compare Bsvd vs Bqr")
  res = compare(Bsvd, Bqr)
  println("")

  Bcqrd = calculate_slice_matrix_chain_qr_dagger(p,l,1,p.slices);
  Bqrd = Bcqrd[1] * diagm(Bcqrd[2]) * Bcqrd[3];
  println("QR consistency: ")
  println("Compare Bqrdagger vs Bqr")
  compare(Bqrd', Bqr)
  # println("")
  # println("Compare Bqrdagger vs Bsvd")
  # compare(Bqrd', Bsvd)

  return res
end
# Worked


function test_greens_svd_vs_qr(p::Parameters, l::Lattice, slice::Int=-1)
  if slice == -1
    slice = rand(1:p.slices-1)
  end
  println("Slice: ", slice)

  @time gqr = calculate_greens_qr(p,l,slice)
  @time gsvd = calculate_greens_udv(p,l,slice)
  println("Compare greens calculation: QR vs SVD")
  compare(gqr, gsvd)
  println("")

  println("Wrapping accuracy:")
  println("QR:")
  gqr2 = calculate_greens_qr(p,l,slice+1)
  gqrw = wrap_greens(p,l,gqr,slice,1)
  compare(gqr2, gqrw)

  println("\nSVD:")
  gsvd2 = calculate_greens_udv(p,l,slice+1)
  gsvdw = wrap_greens(p,l,gsvd,slice,1)
  compare(gsvd2, gsvdw)

end
# Worked


using PyPlot
function plot_svs_of_slice_matrix_chain_qr(p::Parameters, l::Lattice)
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
  a[:set_title]("Stabilize, SVD")

  T = calculate_slice_matrix_chain_qr(p,l,1,p.slices,1)
  svs = T[4]
  c = fig[:add_subplot](133, sharey=b, sharex=b)
  c[:plot](svs[:,:]')
  setp(c[:get_yticklabels](), visible=false)
  c[:set_xlabel]("Inverse temperature \$\\beta\$")
  c[:set_title]("Stabilize, QR")

  # tight_layout()
  subplots_adjust(wspace = 0.0)
  show()
  println(maximum(svs))
  nothing
end