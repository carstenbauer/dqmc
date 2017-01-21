# define parameters type
type parameters
  beta::Float64
  delta_tau::Float64
  slices::Int
  safe_mult::Int
  lattice_file::String
  hsfield::Array{Float64, 3} # dim 1: op component, dim 2: linearized spatial lattice, dim 3: imag time
  thermalization::Int # no measurements, no saving
  measurements::Int # save and (maybe) measure
  mu::Float64
  lambda::Float64
  r::Float64
  u::Float64
  flv::Int # flavors: GF matrix has size flv*l.sites x flv*l.sites
  parameters() = new()
end
