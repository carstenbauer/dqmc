using Distributions

type Parameters
  lattice_file::String
  beta::Float64
  delta_tau::Float64
  slices::Int
  safe_mult::Int

  thermalization::Int # no measurements, no saving
  measurements::Int # save and (maybe) measure

  hsfield::Array{Float64, 3} # dim 1: op component, dim 2: linearized spatial lattice, dim 3: imag time
  boson_action::Float64

  mu::Float64
  lambda::Float64
  r::Float64
  c::Float64
  u::Float64
  flv::Int # flavors: GF matrix has size flv*l.sites x flv*l.sites

  box::Distributions.Uniform{Float64}

  interaction_sinh::Array{Float64, 2}
  interaction_cosh::Array{Float64, 2}

  Parameters() = new()
end