using LightXML
using Dates

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

  box::Float64
  box_global::Float64

  global_updates::Bool
  global_rate::Int64

  sparsity_limit::Float64 # upper sparsity limit for chkr folding heuristic
  chkr::Bool # checkerboard
  Bfield::Bool # artificial magnetic field to reduce finite size effects

  output_file::String
  write_every_nth::Int # measure after 'write_every_nth' up-down sweeps(!)

  all_checks::Bool # true: check for propagation instblts. and S_b consistency

  seed::Int

  resume::Bool
  prethermalized::Int # how many thermal ud-sweeps are captured by custom start configuration 'thermal_init/conf'

  edrun::Bool # if true, only the mass term of the bosonic action is considered
  walltimelimit::Dates.DateTime

  obs::Set{Symbol} # what observables to measure during DQMC (excl. configurations)

  function Params()
    p = new()
    p.global_updates = true
    p.chkr = true
    p.Bfield = false
    p.box = 0.5
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
    p.walltimelimit = Dates.DateTime("2099", "YYYY") # effective infinity
    # p.obs = Set{Symbol}()
    p.obs = Set{Symbol}()
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
  big_delta_tau = parse(BigFloat, params["DELTA_TAU"])
  if haskey(params, "BETA")
    p.slices = Int(parse(BigFloat, params["BETA"])/big_delta_tau)
    println("SLICES = ", p.slices)
  elseif haskey(params, "T")
    p.slices = Int((big"1"/parse(BigFloat, params["T"]))/big_delta_tau)
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
    p.box = parse(Float64, params["BOX_HALF_LENGTH"])
    p.box_global = p.box
  end
  if haskey(params,"BOX_GLOBAL_HALF_LENGTH")
    p.box_global = parse(Float64, params["BOX_GLOBAL_HALF_LENGTH"])
  end
  haskey(params,"GLOBAL_RATE") && (p.global_rate = parse(Int64, params["GLOBAL_RATE"]))
  haskey(params,"WRITE_EVERY_NTH") && (p.write_every_nth = parse(Int64, params["WRITE_EVERY_NTH"]))
  haskey(params, "SPARSITY_LIMIT") && (p.sparsity_limit = parse(Float64, params["SPARSITY_LIMIT"]))
  haskey(params, "EDRUN") && (p.edrun = parse(Bool, lowercase(params["EDRUN"])))

  # observables
  for obs in (:greens, :boson_action)
    obs_str = uppercase(string(obs))
    if haskey(params, obs_str)
      if parse(Bool, lowercase(params[obs_str]))
        push!(p.obs, obs)
      elseif obs in p.obs
        pop!(p.obs, obs)
      end
    end
  end

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
    p.L = parse(Int, p.lattice_file[findlast(isequal('_'), collect(p.lattice_file))+1:end-4])
  else
    hn = lowercase(gethostname())
    lat = ""
    if p.Nhoppings == "none" && p.NNhoppings == "none"
      lat = "square_L_$(p.L)_W_$(p.L).xml"
    else
      lat = "NNsquare_L_$(p.L)_W_$(p.L).xml"
    end

    if "LATTICES" in keys(ENV)
      p.lattice_file = joinpath(ENV["LATTICES"], lat)
    else
      # try to deduce lattice path from hostname
      if occursin("cheops", hn)
        p.lattice_file = "/projects/ag-trebst/bauer/lattices/"*lat
      elseif occursin("fz-juelich", hn) || occursin("juwels", hn)
        p.lattice_file = "/gpfs/homea/hku27/hku273/lattices/"*lat
      elseif occursin("thp", hn)
        p.lattice_file = "/home/bauer/lattices/"*lat
      elseif occursin("thinkable", hn)
        p.lattice_file = "C:/Users/carsten/Desktop/sciebo/lattices/"*lat
      elseif occursin("travis", hn)
        p.lattice_file = joinpath(dirname(dirname(@__FILE__)), "test/lattices/", lat)
      else
        error("Unrecognized host. Can't deduce lattice file path.")
      end
    end
  end
  nothing
end