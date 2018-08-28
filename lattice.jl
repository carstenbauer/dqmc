mutable struct Lattice{H<:Number} # H = HoppingEltype
  dim::Int
  sites::Int
  L::Int
  n_neighbors::Int
  n_bonds::Int
  t::Matrix{Float64} # colidx = flavor/band, rowidx = hor,ver
  tN::Matrix{Float64} # colidx = flavor/band, rowidx = just one (Yoni txy)
  tNN::Matrix{Float64} # colidx = flavor/band, rowidx = hor,ver
  time_neighbors::Matrix{Int} # colidx = slice, rowidx = up, down
  neighbors::Matrix{Int} # colidx = site, rowidx = up down right left
  Nneighbors::Matrix{Int} # ur, dr, dl, ul
  NNneighbors::Matrix{Int} # uu, rr, dd, ll

  bonds::Matrix{Int} # src, trg, type
  bond_vecs::Matrix{Float64}

  peirls::Matrix{Matrix{Float64}} # phis and NOT exp(im*phi); cols = flavor, row = spin; each inner matrix: trg, src (linidx)

  hopping_matrix_exp::Matrix{H} # mu included
  hopping_matrix_exp_inv::Matrix{H}

  chkr_hop_half::Vector{SparseMatrixCSC{H, Int64}}
  chkr_hop_half_inv::Vector{SparseMatrixCSC{H, Int64}}
  chkr_hop_half_dagger::Vector{SparseMatrixCSC{H, Int64}}

  # -- chkr groups folding (regrouping)
  n_folded::Int
  folded::Vector{UnitRange{Int64}}

  chkr_hop_half_folded::Vector{SparseMatrixCSC{H, Int64}}
  chkr_hop_half_inv_folded::Vector{SparseMatrixCSC{H, Int64}}
  chkr_hop_half_dagger_folded::Vector{SparseMatrixCSC{H, Int64}}

  chkr_hop_half_folded_rev::Vector{SparseMatrixCSC{H, Int64}}
  chkr_hop_half_inv_folded_rev::Vector{SparseMatrixCSC{H, Int64}}
  chkr_hop_half_dagger_folded_rev::Vector{SparseMatrixCSC{H, Int64}}
  # --

  chkr_hop::Vector{SparseMatrixCSC{H, Int64}} # without prefactor 0.5 in matrix exponentials
  chkr_hop_inv::Vector{SparseMatrixCSC{H, Int64}}
  chkr_hop_dagger::Vector{SparseMatrixCSC{H, Int64}}
  chkr_mu_half::SparseMatrixCSC{H, Int64}
  chkr_mu_half_inv::SparseMatrixCSC{H, Int64}
  chkr_mu::SparseMatrixCSC{H, Int64}
  chkr_mu_inv::SparseMatrixCSC{H, Int64}

  # Currently NOT used in checkerboard.jl (but generic chkr)
  checkerboard::Matrix{Int} # src, trg, bondid
  groups::Vector{UnitRange}
  n_groups::Int

  Lattice{H}() where H = new{H}()
end

function load_lattice(mc::AbstractDQMC)
  mc.l.L = mc.p.L
  mc.p.hoppings != "none" && (mc.l.t = reshape([parse(Float64, f) for f in split(mc.p.hoppings, ',')],(2,2)))
  mc.p.Nhoppings != "none" && (mc.l.tN = reshape([parse(Float64, f) for f in split(mc.p.Nhoppings, ',')],(1,2)))
  mc.p.NNhoppings != "none" && (mc.l.tNN = reshape([parse(Float64, f) for f in split(mc.p.NNhoppings, ',')],(2,2)))

  init_lattice_from_filename(mc.p.lattice_file, mc.l)
  init_neighbors_table(mc)
  init_time_neighbors_table(mc)
  init_hopping_matrices(mc)
end

function init_lattice_from_filename(filename::String, l::Lattice)
  xdoc = parse_file(filename)
  graph = LightXML.root(xdoc)
  l.sites = 1

  l.sites = parse(Int, attribute(graph, "vertices"; required=true))
  l.dim = parse(Int, attribute(graph, "dimension"; required=true))

  edges = get_elements_by_tagname(graph, "EDGE")
  l.n_bonds = length(edges)

  # bonds & bond vectors
  println("\nLoading lattice with ", l.sites , " sites")
  l.bonds = zeros(l.n_bonds, 4)
  l.bond_vecs = zeros(l.n_bonds, l.dim)
  v = Vector{Float64}(l.dim)
  for (i, edge) in enumerate(edges)
    src = 0
    trg = 0
    src = parse(Int, attribute(edge, "source"; required=true))
    trg = parse(Int, attribute(edge, "target"; required=true))
    typ = parse(Int, attribute(edge, "type"; required=true))
    id = parse(Int, attribute(edge, "id"; required=true))
    v = [parse(Float64, f) for f in split(attribute(edge, "vector"; required=true)," ")]

    if id != i error("Edges in lattice file must be sorted from 1 to N!") end

    l.bonds[i, 1] = src
    l.bonds[i, 2] = trg
    l.bonds[i, 3] = typ
    l.bond_vecs[i, :] = v
  end

  l.n_neighbors = count(x->x==1, l.bonds[:, 1]) + count(x->x==1, l.bonds[:, 2]) # neighbors of site 1
end

function init_neighbors_table(mc::AbstractDQMC)
  const l = mc.l

  println("Initializing neighbor-tables")
  sql = reshape(1:l.sites, l.L, l.L)

  # Nearest neighbors
  if mc.p.hoppings != "none"
    up = circshift(sql,(-1,0))
    right = circshift(sql,(0,-1))
    down = circshift(sql,(1,0))
    left = circshift(sql,(0,1))
    l.neighbors = vcat(up[:]',right[:]',down[:]',left[:]')
  end

  # Next-nearest neighbors (xy, yx)
  if mc.p.Nhoppings != "none"
    ur = circshift(sql,(-1,-1))
    dr = circshift(sql,(1,-1))
    dl = circshift(sql,(1,1))
    ul = circshift(sql,(-1,1))
    l.Nneighbors = vcat(ur[:]',dr[:]',dl[:]',ul[:]')
  end

  # Next-next nearest neighbors (xx, yy)
  if mc.p.NNhoppings != "none"
    uu = circshift(sql,(-2,0))
    rr = circshift(sql,(0,-2))
    dd = circshift(sql,(2,0))
    ll = circshift(sql,(0,2))
    l.NNneighbors = vcat(uu[:]',rr[:]',dd[:]',ll[:]')
  end
  nothing
end


"""
Periodic boundary conditions in imaginary time
"""
function init_time_neighbors_table(mc::AbstractDQMC)
  const l = mc.l
  const p = mc.p

  l.time_neighbors = zeros(Int64, 2, p.slices)
  for s in 1:p.slices
    l.time_neighbors[1,s] = s==p.slices?1:s+1
    l.time_neighbors[2,s] = s==1?p.slices:s-1
  end
end
