if !isdefined(:HoppingType)
  global const HoppingType = Complex128; # assume worst case
  warn("HoppingType wasn't set on loading lattice.jl")
  println("HoppingType = ", HoppingType)
end

using LightXML
using Helpers

# define lattice type
mutable struct Lattice
  dim::Int
  sites::Int
  L::Int
  n_neighbors::Int
  n_bonds::Int
  t::Matrix{Float64} # colidx = flavor/band, rowidx = hor,ver
  time_neighbors::Matrix{Int} # colidx = slice, rowidx = up, down
  neighbors::Matrix{Int} # colidx = site
                           # first = up, second = right, third and fourth not ordered

  bonds::Matrix{Int}
  bond_vecs::Matrix{Float64}
  site_bonds::Matrix{Int}

  peirls::Matrix{Matrix{Float64}} # phis and NOT exp(im*phi); cols = flavor, row = spin; each inner matrix: trg, src (linidx)

  hopping_matrix_exp::Matrix{HoppingType} # mu included
  hopping_matrix_exp_inv::Matrix{HoppingType}

  chkr_hop_half::Vector{SparseMatrixCSC{HoppingType, Int64}}
  chkr_hop_half_inv::Vector{SparseMatrixCSC{HoppingType, Int64}}
  chkr_hop_half_dagger::Vector{SparseMatrixCSC{HoppingType, Int64}}
  chkr_hop::Vector{SparseMatrixCSC{HoppingType, Int64}} # without prefactor 0.5 in matrix exponentials
  chkr_hop_inv::Vector{SparseMatrixCSC{HoppingType, Int64}}
  chkr_hop_dagger::Vector{SparseMatrixCSC{HoppingType, Int64}}
  chkr_mu_half::SparseMatrixCSC{HoppingType, Int64}
  chkr_mu_half_inv::SparseMatrixCSC{HoppingType, Int64}
  chkr_mu::SparseMatrixCSC{HoppingType, Int64}
  chkr_mu_inv::SparseMatrixCSC{HoppingType, Int64}

  # Currently NOT used in checkerboard.jl
  checkerboard::Matrix{Int}
  groups::Vector{UnitRange}

  Lattice() = new()
end

function load_lattice(p::Parameters, l::Lattice)
  l.L = p.L
  l.t = reshape([parse(Float64, f) for f in split(p.hoppings, ',')],(2,2))
  init_lattice_from_filename(p.lattice_file, l)
  init_neighbors_table(p,l)
  init_time_neighbors_table(p,l)
  if p.Bfield
    init_peirls_phases(p,l)
    init_hopping_matrix_exp_Bfield(p,l)
    init_checkerboard_matrices_Bfield(p,l)
  else
    init_hopping_matrix_exp(p,l)
    init_checkerboard_matrices(p,l)
  end
end

function init_lattice_from_filename(filename::String, l::Lattice)
  xdoc = parse_file(filename)
  xroot = LightXML.root(xdoc)
  l.sites = 1

  # Parse header
  for a in attributes(xroot)
      if LightXML.name(a) == "vertices"
        l.sites = parse(Int, value(a))
      elseif LightXML.name(a) == "dimension"
        l.dim = parse(Int, value(a))
      end
    end

  edges = get_elements_by_tagname(xroot, "EDGE")
  l.n_bonds = length(edges)

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
        end
      end

      if edges_used[bd_id] == 1 continue end
      if sites_used[src] == 1 continue end
      if sites_used[trg] == 1 continue end

      if src == 1 || trg == 1
        l.n_neighbors += 1
      end

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

function init_hopping_matrix_exp(p::Parameters,l::Lattice)::Void
  println("Initializing hopping exponentials")
  !p.Bfield || warn("You should be using `init_hopping_matrix_exp_Bfield()` or set p.Bfield = false!")

  Tx = diagm(fill(-p.mu,l.sites))
  Ty = diagm(fill(-p.mu,l.sites))
  hor_nb = [2,4]
  ver_nb = [1,3]

  # Nearest neighbor hoppings
  @inbounds @views begin
    for src in 1:l.sites
      for nb in hor_nb # horizontal neighbors
        trg = l.neighbors[nb,src]
        Tx[trg,src] += -l.t[1,1] # t_x_hor
        Ty[trg,src] += -l.t[1,2] # t_y_hor
      end

      for nb in ver_nb # vertical neighbors
        trg = l.neighbors[nb,src]
        Tx[trg,src] += -l.t[2,1] # t_x_ver
        Ty[trg,src] += -l.t[2,2] # t_y_ver
      end
    end
  end

  eTx_minus = expm(-0.5 * p.delta_tau * Tx)
  eTy_minus = expm(-0.5 * p.delta_tau * Ty)
  eTx_plus = expm(0.5 * p.delta_tau * Tx)
  eTy_plus = expm(0.5 * p.delta_tau * Ty)

  if p.opdim == 3
    l.hopping_matrix_exp = cat([1,2],eTx_minus,eTy_minus,eTx_minus,eTy_minus)
    l.hopping_matrix_exp_inv = cat([1,2],eTx_plus,eTy_plus,eTx_plus,eTy_plus)
  else # O(1) and O(2)
    l.hopping_matrix_exp = cat([1,2],eTx_minus,eTy_minus)
    l.hopping_matrix_exp_inv = cat([1,2],eTx_plus,eTy_plus)
  end

  return nothing
end

function init_peirls_phases(p::Parameters, l::Lattice)
  println("Initializing Peirls phases (Bfield)")
  const L = l.L

  B = zeros(2,2) # colidx = flavor, rowidx = spin up,down
  if p.Bfield
    B[1,1] = B[2,2] = 2 * pi / l.sites
    B[1,2] = B[2,1] = - 2 * pi / l.sites
  end

  l.peirls = Matrix{Matrix{Float64}}(2,2) # colidx = flavor, rowidx = spin up,down
  for f in 1:2 # flv
    for s in 1:2 # spin
      phis = fill(NaN, L, L, L, L)

      for x in 1:L
        for y in 1:L
          xp = mod1(x + 1, L)
          yp = mod1(y + 1, L)
          
          #nn
          phis[x,y,x,yp] = 0
          phis[x,yp,x,y] = 0
          
          phis[x,y,xp,y] = - B[s,f] * (y - 1)
          phis[xp,y,x,y] = - phis[x,y,xp,y]
          if y == L
              phis[x,y,x,yp] = B[s,f] * L * (x -1)
              phis[x,yp,x,y] = - phis[x,y,x,yp]
          end
        end
      end
      l.peirls[s,f] = reshape(permutedims(phis, [2,1,4,3]), (l.sites,l.sites))

    end
  end
end

function init_hopping_matrix_exp_Bfield(p::Parameters,l::Lattice)::Void  
  println("Initializing hopping exponentials (Bfield)")
  p.Bfield || warn("You should be using `init_hopping_matrix_exp()` or set p.Bfield = true!")

  T = Matrix{Matrix{HoppingType}}(2,2) # colidx = flavor, rowidx = spin up,down
  for i in 1:4
    T[i] = convert(Matrix{HoppingType}, diagm(fill(-p.mu,l.sites)))
  end

  hor_nb = [2,4]
  ver_nb = [1,3]

  # Nearest neighbor hoppings
  @inbounds @views begin
    for f in 1:2
      for s in 1:2

        for src in 1:l.sites
          for nb in hor_nb # horizontal neighbors
            trg = l.neighbors[nb,src]
            T[s,f][trg,src] += - exp(im * l.peirls[s,f][trg,src]) * l.t[1,f] # horizontal
          end

          for nb in ver_nb # vertical neighbors
            trg = l.neighbors[nb,src]
            T[s,f][trg,src] += - exp(im * l.peirls[s,f][trg,src]) * l.t[2,f] # vertical
          end
        end

      end
    end
  end

  eT_minus = map(Ti -> expm(-0.5 * p.delta_tau * Ti), T)
  eT_plus = map(Ti -> expm(0.5 * p.delta_tau * Ti), T)

  if p.opdim == 3
    l.hopping_matrix_exp = cat([1,2], eT_minus[1,1], eT_minus[2,2], eT_minus[2,1], eT_minus[1,2])
    l.hopping_matrix_exp_inv = cat([1,2], eT_plus[1,1], eT_plus[2,2], eT_plus[2,1], eT_plus[1,2])
  else # O(1) and O(2)
    l.hopping_matrix_exp = cat([1,2], eT_minus[1,1], eT_minus[2,2])
    l.hopping_matrix_exp_inv = cat([1,2], eT_plus[1,1], eT_plus[2,2])
  end

  return nothing
end


function init_neighbors_table(p::Parameters,l::Lattice)
  println("Initializing neighbor-tables")
  sql = reshape(1:l.sites, l.L, l.L)

  # Nearest neighbors
  up = circshift(sql,(-1,0))
  right = circshift(sql,(0,-1))
  down = circshift(sql,(1,0))
  left = circshift(sql,(0,1))
  l.neighbors = vcat(up[:]',right[:]',down[:]',left[:]')

  nothing
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
