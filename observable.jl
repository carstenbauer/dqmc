import HDF5

type Observable{T<:Union{Number,AbstractArray}}
  name::String
  count::Int
  timeseries::Array{T, 1}

  Observable(name::String,alloc::Int) = new(name,0,Array{T, 1}(alloc))
  Observable(name::String) = Observable{T}(name,256)
end


function add_value{T}(obs::Observable{T}, value::T)
  if length(obs.timeseries) < obs.count+1
    # backup if allocation wasn't correct
    push!(obs.timeseries,copy(value))
  end
  obs.timeseries[obs.count+1] = copy(value)
  obs.count += 1
end


function obs2hdf5{T}(filename::String, obs::Observable{T})
  HDF5.h5write(filename, "simulation/results/" * obs.name * "/count", obs.count)
  timeseries_matrix = cat(ndims(obs.timeseries[1])+1,obs.timeseries[1:obs.count]...)
  nd = ndims(timeseries_matrix)
  if eltype(timeseries_matrix)<:Real
    HDF5.h5write(filename, "simulation/results/" * obs.name * "/timeseries", timeseries_matrix)
    HDF5.h5write(filename, "simulation/results/" * obs.name * "/mean", squeeze(mean(timeseries_matrix,nd),nd))
  else
    HDF5.h5write(filename, "simulation/results/" * obs.name * "/timeseries_real", real(timeseries_matrix))
    HDF5.h5write(filename, "simulation/results/" * obs.name * "/timeseries_imag", imag(timeseries_matrix))
    HDF5.h5write(filename, "simulation/results/" * obs.name * "/mean_real", squeeze(mean(real(timeseries_matrix),nd),nd))
    HDF5.h5write(filename, "simulation/results/" * obs.name * "/mean_imag", squeeze(mean(imag(timeseries_matrix),nd),nd))
  end
end

function load_obs(filename::String, obsname::String)
  # TODO: load_obs from hdf5: Check if complex or real, and load timeseries.
end