using FFTW # v.0.6 naming bug
using Distributions
using Git
include("xml_parameters.jl")

mutable struct Parameters
  lattice_file::String
  beta::Float64
  delta_tau::Float64
  slices::Int
  safe_mult::Int
  opdim::Int # order parameter dimension

  thermalization::Int # no measurements, no saving
  measurements::Int # save and (maybe) measure

  hsfield::Array{Float64, 3} # dim 1: op component, dim 2: linearized spatial lattice, dim 3: imag time
  boson_action::Float64

  mu::Float64
  lambda::Float64
  r::Float64
  c::Float64
  u::Float64
  flv::Int # flavors: GF matl_sites size flv*l.sites x flv*l.sites

  box::Distributions.Uniform{Float64}
  box_global::Distributions.Uniform{Float64}

  global_updates::Bool
  global_rate::Int64

  chkr::Bool # checkerboard
  Bfield::Bool # artificial magnetic field to reduce finite size effects

  output_file::String
  write_every_nth::Int

  all_checks::Bool # true: check for propagation instblts. and S_b consistency

  function Parameters()
    p = new()
    p.global_updates = true
    p.chkr = true
    p.Bfield = false
    p.box = Uniform(-0.5,0.5)
    p.box_global = Uniform(-0.1,0.1)
    p.global_rate = 5
    p.write_every_nth = 1
    p.all_checks = false
    p.opdim = 3
    return p
  end
end


function parse_inputxml(p::Parameters, input_xml::String)
  # READ INPUT XML
  params = Dict{Any, Any}()
  try
    params = xml2parameters(input_xml)

    # Check and store code version (git commit)
    if haskey(params,"GIT_COMMIT_DQMC") && Git.head(dir=dirname(@__FILE__)) != params["GIT_COMMIT_DQMC"]
      warn("Git commit in input xml file does not match current commit of code.")
    end
    params["GIT_COMMIT_DQMC"] = Git.head(dir=dirname(@__FILE__))

    parameters2hdf5(params, output_file)
  catch e
    println(e)
  end

  ### PARSE MANDATORY PARAMS
  p.thermalization = parse(Int, params["THERMALIZATION"])
  p.measurements = parse(Int, params["MEASUREMENTS"])
  p.slices = parse(Int, params["SLICES"])
  p.delta_tau = parse(Float64, params["DELTA_TAU"])
  p.safe_mult = parse(Int, params["SAFE_MULT"])
  p.lattice_file = params["LATTICE_FILE"]
  p.mu = parse(Float64, params["MU"])
  p.lambda = parse(Float64, params["LAMBDA"])
  p.r = parse(Float64, params["R"])
  p.c = parse(Float64, params["C"])
  p.u = parse(Float64, params["U"])

  p.beta = p.slices * p.delta_tau
  p.flv = 4

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

  if haskey(params, "SEED")
    srand(parse(Int, params["SEED"]));
  end

  if haskey(params,"GLOBAL_UPDATES")
    p.global_updates = parse(Bool, lowercase(params["GLOBAL_UPDATES"]))
  end

  if haskey(params,"CHECKERBOARD")
    p.chkr = parse(Bool, lowercase(params["CHECKERBOARD"]))
  end

  if haskey(params,"B_FIELD")
    p.Bfield = parse(Bool, lowercase(params["B_FIELD"]))
  end


  if haskey(params,"BOX_HALF_LENGTH")
    len = parse(Float64, params["BOX_HALF_LENGTH"])
    p.box = Uniform(-len,len)
  end

  if haskey(params,"BOX_GLOBAL_HALF_LENGTH")
    len = parse(Float64, params["BOX_GLOBAL_HALF_LENGTH"])
    p.box_global = Uniform(-len,len)
  end

  if haskey(params,"GLOBAL_RATE")
    p.global_rate = parse(Int64, params["GLOBAL_RATE"])
  end

  if haskey(params,"WRITE_EVERY_NTH")
    p.write_every_nth = parse(Int64, params["WRITE_EVERY_NTH"])
  end

  params
end