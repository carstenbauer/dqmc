using LightXML
using Helpers

# define hoppings type
type Hoppings
  xh::Float64
  xv::Float64
  yh::Float64
  yv::Float64
end
str(t::Hoppings) = "xh = $(t.xh), xv = $(t.xv), yh = $(t.yh), yv = $(t.yv)"

# define lattice type
type Lattice
  dim::Int
  sites::Int
  L::Int
  n_neighbors::Int
  n_bonds::Int
  t::Hoppings
  time_neighbors::Array{Int, 2} # colidx = slice, rowidx = up, down
  neighbors::Array{Int, 2} # colidx = site
                           # first = up, second = right, third and fourth not ordered
  bonds::Array{Int, 2}
  bond_vecs::Array{Float64, 2}
  site_bonds::Array{Int, 2}

  hopping_matrix_exp::Array{Float64, 2} # mu included
  hopping_matrix_exp_inv::Array{Float64, 2}

  chkr_hop::Array{SparseMatrixCSC, 1}
  chkr_hop_inv::Array{SparseMatrixCSC, 1}
  chkr_mu::SparseMatrixCSC{Float64, Int64}
  chkr_mu_inv::SparseMatrixCSC{Float64, Int64}

  # peter remnants
  checkerboard::Array{Int, 2}
  groups::Array{UnitRange, 1}
  free_fermion_wavefunction::Array{Float64, 2}
  temp_thin::Array{Complex{Float64}, 2}
  temp_square::Array{Complex{Float64}, 2}
  temp_diag::Array{Float64, 1}
  temp_small::Array{Complex{Float64}, 2}

  Lattice() = new()
end

# TODO: checkerboard content in this function
# TODO: too general for square lattice! leave it that way?
function init_lattice_from_filename(filename::String, l::Lattice)
  xdoc = parse_file(filename)
  xroot = LightXML.root(xdoc)
  l.sites = 1

  for a in attributes(xroot)
      if LightXML.name(a) == "vertices"
        l.sites = parse(Int, value(a))
      elseif LightXML.name(a) == "dimension"
        l.dim = parse(Int, value(a))
      end
    end

  edges = get_elements_by_tagname(xroot, "EDGE")
  l.n_bonds = length(edges)

  # checkerboard
  # l.chkr_hop = SparseMatrixCSC[]
  # l.chkr_hop_inv = SparseMatrixCSC[]
  l.groups = UnitRange[]
  l.n_neighbors = 0
  edges_used = zeros(Int64, length(edges))
  l.checkerboard = zeros(3, length(edges))
  group_start = 1
  group_end = 1

  while minimum(edges_used) == 0
    sites_used = zeros(Int64, l.sites)

    for e in edges
      src = 0
      trg = 0
      bd_type = 0
      bd_id = 0

      for a = attributes(e)
        if LightXML.name(a) == "source"
          src = parse(Int, value(a))
        elseif LightXML.name(a) == "target"
          trg = parse(Int, value(a))
        elseif LightXML.name(a) == "type"
          bd_type = parse(Int, value(a)) + 1
        elseif LightXML.name(a) == "id"
          bd_id = parse(Int, value(a))
        end end

      if edges_used[bd_id] == 1 continue end
      if sites_used[src] == 1 continue end
      if sites_used[trg] == 1 continue end

      if src == 1 || trg == 1
        # println("Bond ", src, " - ", trg)
        l.n_neighbors += 1 end

      edges_used[bd_id] = 1
      sites_used[src] = 1
      sites_used[trg] = 1

      l.checkerboard[:, group_end] = [src, trg, bd_type]
      group_end += 1
    end
    push!(l.groups, group_start:group_end-1)
    group_start = group_end
  end

  # bonds & bond vectors
  println("\nLoading lattice with ", l.sites , " sites")
  l.bonds = zeros(l.n_bonds, 2)
  l.bond_vecs = zeros(l.n_bonds, l.dim)
  l.site_bonds = zeros(l.sites, l.n_neighbors)
  for (e_idx, e) in enumerate(edges)
    src = 0
    trg = 0
    vec = Vector{Float64}(l.dim)
    for a = attributes(e)
      if LightXML.name(a) == "source"
        src = parse(Int, value(a))
      elseif LightXML.name(a) == "target"
        trg = parse(Int, value(a))
      elseif LightXML.name(a) == "vector"
        vec = [parse(Float64, f) for f in split(value(a)," ")]
      end
    end

    l.bonds[e_idx, 1] = src
    l.bonds[e_idx, 2] = trg
    l.bond_vecs[e_idx, :] = vec

    idx = findfirst(l.site_bonds[src, :], 0)
    l.site_bonds[src, idx] = e_idx
    idx = findfirst(l.site_bonds[trg, :], 0)
    l.site_bonds[trg, idx] = e_idx
  end

  for i in 1:l.sites
    if findfirst(l.site_bonds[i, :], 0) != 0
      println(i)
      println(l.site_bonds[i, :])
      println(findfirst(l.site_bonds[i, :]))
      error("l.site_bonds is not correctly setup")
    end
  end
end


function init_hopping_matrix_exp(p::Parameters,l::Lattice)
  Tx = diagm(fill(-p.mu,l.sites))
  Ty = diagm(fill(-p.mu,l.sites))
  for b in 1:l.n_bonds
    src = l.bonds[b,1]
    trg = l.bonds[b,2]
    if l.bond_vecs[b,1] == 1
      Tx[trg,src] = Tx[src,trg] = -l.t.xh
      Ty[trg,src] = Ty[src,trg] = -l.t.yh
    else
      Tx[trg,src] = Tx[src,trg] = -l.t.xv
      Ty[trg,src] = Ty[src,trg] = -l.t.yv
    end
  end

  eTx_minus = expm(-0.5 * p.delta_tau * Tx)
  eTy_minus = expm(-0.5 * p.delta_tau * Ty)
  eTx_plus = expm(0.5 * p.delta_tau * Tx)
  eTy_plus = expm(0.5 * p.delta_tau * Ty)
  l.hopping_matrix_exp = cat([1,2],eTx_minus,eTy_minus,eTx_minus,eTy_minus)
  l.hopping_matrix_exp_inv = cat([1,2],eTx_plus,eTy_plus,eTx_plus,eTy_plus)
end


# Checkerboard: Assaad four site version for square lattice
function build_four_site_hopping_matrix(p::Parameters,l::Lattice,corner::Int,flv::String="x")
  sites_clockwise = [corner, l.neighbors[1,corner], l.neighbors[1,l.neighbors[2,corner]], l.neighbors[2,corner]]
  shift_sites_clockwise = circshift(sites_clockwise,-1)
  hop = zeros(8)
  if flv=="y"
    hop = -1 * repeat([l.t.yv, l.t.yh, l.t.yv, l.t.yh], outer=2)
  else
    hop = -1 * repeat([l.t.xv, l.t.xh, l.t.xv, l.t.xh], outer=2)
  end
  return sparse([sites_clockwise; shift_sites_clockwise],[shift_sites_clockwise; sites_clockwise],hop,l.sites,l.sites)
end

function init_checkerboard_matrices(p::Parameters,l::Lattice)
  pc = Int(floor(l.sites/4))

  A_corners = zeros(Int, pc)
  idx = 1
  tolinidx = reshape(collect(1:l.sites),l.L,l.L)
  for x in 1:2:l.L
      for y in 1:2:l.L
          A_corners[idx] = tolinidx[y,x]
          idx += 1
      end
  end
  B_corners = l.neighbors[1,l.neighbors[2,A_corners]]

  Tx_As = Array{SparseMatrixCSC,1}(pc)
  Tx_Bs = Array{SparseMatrixCSC,1}(pc)
  Ty_As = Array{SparseMatrixCSC,1}(pc)
  Ty_Bs = Array{SparseMatrixCSC,1}(pc)

  for c in 1:length(A_corners)
    Tx_As[c] = build_four_site_hopping_matrix(p, l, A_corners[c], "x")
    Ty_As[c] = build_four_site_hopping_matrix(p, l, A_corners[c], "y")
    Tx_Bs[c] = build_four_site_hopping_matrix(p, l, B_corners[c], "x")
    Ty_Bs[c] = build_four_site_hopping_matrix(p, l, B_corners[c], "y")
  end

  Tx_A = sum(Tx_As)
  Ty_A = sum(Ty_As)
  Tx_B = sum(Tx_Bs)
  Ty_B = sum(Ty_Bs)

  eTx_A_minus = sparse(expm(-0.5 * p.delta_tau * full(Tx_A)))
  eTy_A_minus = sparse(expm(-0.5 * p.delta_tau * full(Ty_A)))
  eTx_B_minus = sparse(expm(-0.5 * p.delta_tau * full(Tx_B)))
  eTy_B_minus = sparse(expm(-0.5 * p.delta_tau * full(Ty_B)))
  eTx_A_plus = sparse(expm(0.5 * p.delta_tau * full(Tx_A)))
  eTy_A_plus = sparse(expm(0.5 * p.delta_tau * full(Ty_A)))
  eTx_B_plus = sparse(expm(0.5 * p.delta_tau * full(Tx_B)))
  eTy_B_plus = sparse(expm(0.5 * p.delta_tau * full(Ty_B)))

  # OPT: many <1e-15 entries in expm(-0.5 * p.delta_tau * full(Tx_B))
  # sparsity is therefore not optimal. Can we just set them zero? Check with Mathematica
  # if they should be analytically equal zero.
  #
  # low_cutoff(X::Array, c::Float64) = map(e->e<c?0:e,X)
  # eTx_A_minus = sparse(low_cutoff(expm(-0.5 * p.delta_tau * full(Tx_A)), 1e-15))
  # eTy_A_minus = sparse(low_cutoff(expm(-0.5 * p.delta_tau * full(Ty_A)), 1e-15))
  # eTx_B_minus = sparse(low_cutoff(expm(-0.5 * p.delta_tau * full(Tx_B)), 1e-15))
  # eTy_B_minus = sparse(low_cutoff(expm(-0.5 * p.delta_tau * full(Ty_B)), 1e-15))
  # eTx_A_plus = sparse(low_cutoff(expm(0.5 * p.delta_tau * full(Tx_A)), 1e-15))
  # eTy_A_plus = sparse(low_cutoff(expm(0.5 * p.delta_tau * full(Ty_A)), 1e-15))
  # eTx_B_plus = sparse(low_cutoff(expm(0.5 * p.delta_tau * full(Tx_B)), 1e-15))
  # eTy_B_plus = sparse(low_cutoff(expm(0.5 * p.delta_tau * full(Ty_B)), 1e-15))

  # chemical potential
  # eTmu_minus = spdiagm(fill(exp(-0.5 * p.delta_tau * -p.mu), l.sites))
  # eTmu_plus = spdiagm(fill(exp(0.5 * p.delta_tau * -p.mu), l.sites))

  # not-sparse matrix eTx_A_minus * eTx_B_minus * eTmu_minus is equal to eTx_minus above (non-chkr hopping matrix)
  # individual eTx_A_minus have sparsity ~0.75

  eT_A_minus = cat([1,2],eTx_A_minus,eTy_A_minus,eTx_A_minus,eTy_A_minus) # sparsity > 90%
  eT_B_minus = cat([1,2],eTx_B_minus,eTy_B_minus,eTx_B_minus,eTy_B_minus)
  eT_A_plus = cat([1,2],eTx_A_plus,eTy_A_plus,eTx_A_plus,eTy_A_plus)
  eT_B_plus = cat([1,2],eTx_B_plus,eTy_B_plus,eTx_B_plus,eTy_B_plus)

  l.chkr_hop = [eT_A_minus, eT_B_minus]
  l.chkr_hop_inv = [eT_A_plus, eT_B_plus]
  l.chkr_mu = spdiagm(fill(exp(-0.5 * p.delta_tau * -p.mu), p.flv * l.sites))
  l.chkr_mu_inv = spdiagm(fill(exp(0.5 * p.delta_tau * -p.mu), p.flv * l.sites))
end


function init_neighbors_table(p::Parameters,l::Lattice)
  # OPT: neighbor table: sort down and left neighbors
  l.neighbors = zeros(Int64, 4, l.sites) # unsorted order of neighbors
  for i in 1:l.sites
    l.neighbors[:,i] = filter(x->x!=i,l.bonds[l.site_bonds[i,:],:])
  end
  swap_rows!(l.neighbors,3,1)
  swap_rows!(l.neighbors,4,2)
end


"""
Periodic boundary conditions in imaginary time
"""
function init_time_neighbors_table(p::Parameters,l::Lattice)
  l.time_neighbors = zeros(Int64, 2, p.slices)
  for s in 1:p.slices
    l.time_neighbors[1,s] = s==p.slices?1:s+1
    l.time_neighbors[2,s] = s==1?p.slices:s-1
  end
end
