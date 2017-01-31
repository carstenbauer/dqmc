import HDF5

type observable{T<:Union{Number,Array}}
  name::String
  count::Int
  timeseries::Array{T, 1}

  function observable(name::String,alloc::Int)
    if T <: Number
      new(name,0,zeros(T,alloc))
    elseif T <: Array
      new(name,0,Array{T, 1}(alloc))
    end
  end

  observable(name::String) = observable{T}(name,256)
end


function add_value{T}(obs::observable{T}, value::T)
  if size(obs.timeseries)[1] < obs.count+1
    # backup if allocation wasn't correct
    push!(obs.timeseries,copy(value))
  end
  obs.timeseries[obs.count+1] = copy(value)
  obs.count += 1
end

function obs2hdf5{T}(filename::String, obs::observable{T})
  HDF5.h5write(filename, "simulation/results/" * obs.name * "/count", obs.count - 1)
  HDF5.h5write(filename, "simulation/results/" * obs.name * "/timeseries", obs.timeseries)
  HDF5.h5write(filename, "simulation/results/" * obs.name * "/mean", mean(obs.timeseries))
end
