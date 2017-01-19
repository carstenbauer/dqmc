using LightXML

# define hoppings type
type hoppings
  xh::Float64
  xv::Float64
  yh::Float64
  yv::Float64
end

# define lattice type
type lattice
  sites::Int
  L::Int
  n_neighbors::Int
  n_bonds::Int
  t::hoppings
  urneighbors::Array{Int, 2} # different cols contain up and right neighbors of site=colidx

  checkerboard::Array{Int, 2}
  groups::Array{UnitRange, 1}
  chkr_hop::Array{SparseMatrixCSC, 1}
  chkr_hop_inv::Array{SparseMatrixCSC, 1}
  free_fermion_wavefunction::Array{Float64, 2}

  temp_thin::Array{Complex{Float64}, 2}
  temp_square::Array{Complex{Float64}, 2}
  temp_diag::Array{Float64, 1}
  temp_small::Array{Complex{Float64}, 2}

  bonds::Array{Int, 2}
  site_bonds::Array{Int, 2}

  lattice() = new()
end


function init_lattice_from_filename(filename::String, l::lattice)
  l.chkr_hop = SparseMatrixCSC[]
  l.chkr_hop_inv = SparseMatrixCSC[]
  l.groups = UnitRange[]
  l.n_neighbors = 0

  xdoc = parse_file(filename)
  xroot = LightXML.root(xdoc)
  l.sites = 1

  for a in attributes(xroot)
      if LightXML.name(a) == "vertices"
          l.sites = parse(Int, value(a))
      end
    end

  edges = get_elements_by_tagname(xroot, "EDGE")
  l.n_bonds = length(edges)

  edges_used = zeros(Int64, length(edges))
  edge_run = 0

  l.checkerboard = zeros(3, length(edges))

  group_start = 1
  group_end = 1

  while minimum(edges_used) == 0 && edge_run < 100
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
        println("Bond ", src, " - ", trg)
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

  println("Initializing latice with ", l.n_neighbors, " neighbors")
  l.bonds = zeros(l.n_bonds, 2)
  l.site_bonds = zeros(l.sites, l.n_neighbors)
  for (e_idx, e) in enumerate(edges)
    src = 0
    trg = 0
    for a = attributes(e)
      if LightXML.name(a) == "source"
        src = parse(Int, value(a))
      elseif LightXML.name(a) == "target"
        trg = parse(Int, value(a))
      end
    end

    l.bonds[e_idx, 1] = src
    l.bonds[e_idx, 2] = trg

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
