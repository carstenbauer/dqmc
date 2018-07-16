using FFTW # v.0.6 naming bug
using Distributions

mutable struct Params
  lattice_file::String
  hoppings::String # nearest neighbor
  Nhoppings::String # next nearest neighbor
  NNhoppings::String # next next nearest neighbor
  beta::Float64
  delta_tau::Float64
  L::Int
  slices::Int
  safe_mult::Int
  opdim::Int # order parameter dimension

  thermalization::Int # no measurements, no saving
  measurements::Int # save and (maybe) measure

  hsfield::Array{Float64, 3} # dim 1: op component, dim 2: linearized spatial lattice, dim 3: imag time
  boson_action::Float64

  mu1::Float64 # x flavor
  mu2::Float64 # y flavor
  lambda::Float64
  r::Float64
  c::Float64
  u::Float64
  flv::Int # flavors: GF matl_sites size flv*l.sites x flv*l.sites

  box::Distributions.Uniform{Float64}
  box_global::Distributions.Uniform{Float64}

  global_updates::Bool
  global_rate::Int64

  sparsity_limit::Float64 # upper sparsity limit for chkr folding heuristic
  chkr::Bool # checkerboard
  Bfield::Bool # artificial magnetic field to reduce finite size effects

  output_file::String
  write_every_nth::Int

  all_checks::Bool # true: check for propagation instblts. and S_b consistency

  seed::Int

  resume::Bool
  prethermalized::Int # how many thermal ud-sweeps are captured by custom start configuration 'thermal_init/conf'

  edrun::Bool # if true, only the mass term of the bosonic action is considered

  function Params()
    p = new()
    p.global_updates = true
    p.chkr = true
    p.Bfield = false
    p.box = Uniform(-0.5,0.5)
    p.box_global = p.box
    p.global_rate = 5
    p.write_every_nth = 1
    p.all_checks = true
    p.opdim = 3
    p.flv = 4
    p.seed = 4729339882041979125
    p.resume = false
    p.prethermalized = 0
    p.sparsity_limit = 0.01
    p.hoppings = "none"
    p.Nhoppings = "none"
    p.NNhoppings = "none"
    p.edrun = false
    return p
  end
end

"""
    set_parameters(p::Params, params::Dict)

params::Dict -> p::Params (mandatory and optional).
Also triggers `deduce_remaining_parameters()` to e.g. calculate p.beta and set global DataTypes.
"""
function set_parameters(p::Params, params::Dict)
  ### PARSE MANDATORY PARAMS
  if haskey(params, "WARMUP")
    p.thermalization = parse(Int, params["WARMUP"])/2
  else
    p.thermalization = parse(Int, params["THERMALIZATION"])
  end
  if haskey(params, "SWEEPS")
    p.measurements = parse(Int, params["SWEEPS"])/2
  else
    p.measurements = parse(Int, params["MEASUREMENTS"])
  end
  p.delta_tau = parse(Float64, params["DELTA_TAU"])
  if haskey(params, "BETA")
    p.slices = Int(parse(Float64, params["BETA"])/p.delta_tau)
    println("SLICES = ", p.slices)
  elseif haskey(params, "T")
    p.slices = Int((1/parse(Float64, params["T"]))/p.delta_tau)
    println("SLICES = ", p.slices)
  else
    p.slices = parse(Int, params["SLICES"])
  end
  p.safe_mult = parse(Int, params["SAFE_MULT"])
  p.lattice_file = ""
  if haskey(params, "LATTICE_FILE")
    p.lattice_file = params["LATTICE_FILE"]
  else
    p.L = parse(Int, params["L"])
  end

  if haskey(params, "HOPPINGS")
    s = params["HOPPINGS"]
    sum(abs.(parse.(Float64, split(s, ",")))) > 0 && (p.hoppings = params["HOPPINGS"])
  end
  if haskey(params, "N-HOPPINGS")
    s = params["N-HOPPINGS"]
    sum(abs.(parse.(Float64, split(s, ",")))) > 0 && (p.Nhoppings = params["N-HOPPINGS"])
  end
  if haskey(params, "NN-HOPPINGS")
    s = params["NN-HOPPINGS"]
    sum(abs.(parse.(Float64, split(s, ",")))) > 0 && (p.NNhoppings = params["NN-HOPPINGS"])
  end
  if haskey(params, "MU1") && haskey(params, "MU2")
    p.mu1 = parse(Float64, params["MU1"])
    p.mu2 = parse(Float64, params["MU2"])
  elseif haskey(params, "MU1")
    p.mu1 = parse(Float64, params["MU1"])
    p.mu2 = parse(Float64, params["MU1"])
  else
    p.mu1 = parse(Float64, params["MU"])
    p.mu2 = parse(Float64, params["MU"])
  end
  
  p.lambda = parse(Float64, params["LAMBDA"])
  p.r = parse(Float64, params["R"])
  p.c = parse(Float64, params["C"])
  p.u = parse(Float64, params["U"])
  ###

  ### PARSE OPTIONAL PARAMS
  if haskey(params, "OPDIM")
    p.opdim = parse(Int, params["OPDIM"]);
    if p.opdim == 1 || p.opdim == 2
      p.flv = 2
    elseif p.opdim == 3
      p.flv = 4
    else
      error("OPDIM must be 1, 2 or 3!")
    end
  end
  haskey(params, "SEED") && (p.seed = parse(Int, params["SEED"]))
  haskey(params,"GLOBAL_UPDATES") && (p.global_updates = parse(Bool, lowercase(params["GLOBAL_UPDATES"])))
  haskey(params,"CHECKERBOARD") && (p.chkr = parse(Bool, lowercase(params["CHECKERBOARD"])))
  haskey(params,"BFIELD") && (p.Bfield = parse(Bool, lowercase(params["BFIELD"])))
  if haskey(params,"BOX_HALF_LENGTH")
    len = parse(Float64, params["BOX_HALF_LENGTH"])
    p.box = Uniform(-len,len)
    p.box_global = p.box
  end
  if haskey(params,"BOX_GLOBAL_HALF_LENGTH")
    len = parse(Float64, params["BOX_GLOBAL_HALF_LENGTH"])
    p.box_global = Uniform(-len,len)
  end
  haskey(params,"GLOBAL_RATE") && (p.global_rate = parse(Int64, params["GLOBAL_RATE"]))
  haskey(params,"WRITE_EVERY_NTH") && (p.write_every_nth = parse(Int64, params["WRITE_EVERY_NTH"]))
  haskey(params, "SPARSITY_LIMIT") && (p.sparsity_limit = parse(Float64, params["SPARSITY_LIMIT"]))
  haskey(params, "EDRUN") && (p.edrun = parse(Bool, lowercase(params["EDRUN"])))

  deduce_remaining_parameters(p)

  nothing
end

"""
    deduce_remaining_parameters(p::Params)
    
Assumes that `p` has been loaded from XML or HDF5 and sets remaining (dependent) fields in `p`.
"""
function deduce_remaining_parameters(p::Params)
  p.beta = p.slices * p.delta_tau
  p.hsfield = zeros(p.opdim, 1, p.slices) # just to initialize it somehow

  if p.lattice_file != ""
    p.L = parse(Int, p.lattice_file[findlast(collect(p.lattice_file), '_')+1:end-4])
  else
    hn = lowercase(gethostname())
    lat = ""
    if p.Nhoppings == "none" && p.NNhoppings == "none"
      lat = "square_L_$(p.L)_W_$(p.L).xml"
    else
      lat = "NNsquare_L_$(p.L)_W_$(p.L).xml"
    end
    if contains(hn, "cheops")
      p.lattice_file = "/projects/ag-trebst/bauer/lattices/"*lat
    elseif contains(hn, "fz-juelich")
      p.lattice_file = "/gpfs/homea/hku27/hku273/lattices/"
    elseif contains(hn, "thp")
      p.lattice_file = "/home/bauer/lattices/"*lat
    elseif contains(hn, "thinkable")
      p.lattice_file = "C:/Users/carsten/Desktop/sciebo/lattices/"*lat
    else
      error("Unrecognized host. Can't deduce lattice file path.")
    end
  end
  nothing
end

include("xml_parameters.jl")
include("hdf5_parameters.jl")