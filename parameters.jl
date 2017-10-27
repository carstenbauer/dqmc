using FFTW # v.0.6 naming bug
using Distributions

mutable struct Parameters
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
  flv::Int # flavors: GF matl_sites size flv*l.sites x flv*l.sites

  box::Distributions.Uniform{Float64}
  box_global::Distributions.Uniform{Float64}

  global_updates::Bool
  global_rate::Int64

  chkr::Bool # checkerboard
  Bfield::Bool # artificial magnetic field to reduce finite size effects

  output_file::String
  write_every_nth::Int

  #### Array allocations
  eV::Matrix{Complex128}
  eVop1::Matrix{Complex128}
  eVop2::Matrix{Complex128}

  Parameters() = new()
end


function preallocate_arrays(p::Parameters, l_sites::Int)
  p.eV = zeros(Complex128, p.flv * l_sites, p.flv * l_sites)
  p.eVop1 = zeros(Complex128, p.flv, p.flv)
  p.eVop2 = zeros(Complex128, p.flv, p.flv)

  p.Mtmp = eye(Complex128, p.flv * l_sites)
  nothing
end