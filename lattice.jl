using LightXML
using Helpers

# define lattice type
type Lattice
  dim::Int
  sites::Int
  L::Int
  n_neighbors::Int
  n_bonds::Int
  t::Array{Float64, 2} # colidx = flavor, rowidx = hor,ver
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
      Tx[trg,src] = Tx[src,trg] = -l.t[1,1]
      Ty[trg,src] = Ty[src,trg] = -l.t[1,2]
    else
      Tx[trg,src] = Tx[src,trg] = -l.t[2,1]
      Ty[trg,src] = Ty[src,trg] = -l.t[2,2]
    end
  end

  eTx_minus = expm(-0.5 * p.delta_tau * Tx)
  eTy_minus = expm(-0.5 * p.delta_tau * Ty)
  eTx_plus = expm(0.5 * p.delta_tau * Tx)
  eTy_plus = expm(0.5 * p.delta_tau * Ty)
  l.hopping_matrix_exp = cat([1,2],eTx_minus,eTy_minus,eTx_minus,eTy_minus)
  l.hopping_matrix_exp_inv = cat([1,2],eTx_plus,eTy_plus,eTx_plus,eTy_plus)
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
