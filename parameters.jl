# define parameters type
type parameters
  beta::Float64
  delta_tau::Float64
  slices::Int
  safe_mult::Int
  lattice_path::String
  hsfield::Array{Float64, 2} # dim 1: linearized spatial lattice, dim 2: imag time
  thermalization::Int # no measurements, no saving
  measurements::Int # save and (maybe) measure

  mu::Float64
  lambda::Float64
  r::Float64
  u::Float64

  parameters() = new()
end
