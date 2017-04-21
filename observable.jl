using HDF5

type Observable{T<:Number}
  name::String
  count::Int
  timeseries::Array{T}
  elsize::Tuple{Vararg{Int}}
  eldims::Int
  ellength::Int
  alloc::Int

  Observable(name::String, elsize::Tuple{Vararg{Int}}, alloc::Int) = new(name,0,zeros(T, elsize..., alloc), elsize, ndims(Array{Int}(elsize...)), typeof(elsize)==Tuple{}?1:*(elsize...), alloc)
  Observable(name::String, elsize::Int, alloc::Int) = new(name,0,zeros(T, elsize, alloc), (elsize,), 1, elsize, alloc)
  Observable(name::String, elsize::Tuple{Vararg{Int}}) = Observable{T}(name, elsize, 100)

  Observable(name::String, alloc::Int) = Observable{T}(name,(),alloc)
  Observable(name::String) = Observable{T}(name,100)
end


function add_element{T}(obs::Observable{T}, element::Union{Number,Array})
  if size(element) != obs.elsize
    error("Element size incompatible with observable size.")
  end

  # type conversion, e.g. int -> float
  if eltype(element) != T
    try
      if T<:Number  add_element(obs, convert(T, element))
      else  add_element(obs, convert(Array{T}, element)) end
    catch error("Element dtype not compatible with observable dtype.") end
    return
  end

  if length(obs.timeseries) < obs.count+1
    if length(obs.timeseries) == obs.count info("Exceeding time series preallocation of observable.") end
    obs.timeseries = cat(obs.eldims+1, obs.timeseries, element)
    obs.count += 1
  else
    obs.timeseries[obs.count*obs.ellength+1:(obs.count+1)*obs.ellength] = isa(element, Number)?element:element[:]
    obs.count += 1
  end
end


function obs2hdf5{T}(filename::String, obs::Observable{T})
  if isempty(obs) return nothing end

  if obs.count != obs.alloc
    warn("Saving incomplete time series chunk of observable \"$obs.name\".")
  end

  h5open(filename, isfile(filename)?"r+":"w") do f

    new_count = -1
    if !exists(f, "obs/" * obs.name)
      # initialze chunk storage
      write(f, "obs/" * obs.name * "/count", obs.count)
      if T<:Real
        d_create(f, "obs/" * obs.name * "/timeseries", T, ((obs.elsize...,obs.count),(obs.elsize...,-1)), "chunk", (obs.elsize...,obs.alloc), "compress", 9)

        # if obs.eldims == 0
        write(f, "obs/" * obs.name * "/mean", mean(obs.timeseries, ndims(obs.timeseries)))
        # end
      else
        d_create(f, "obs/" * obs.name * "/timeseries_real", T.types[1], ((obs.elsize...,obs.count),(obs.elsize...,-1)), "chunk", (obs.elsize...,obs.alloc), "compress", 9)
        d_create(f, "obs/" * obs.name * "/timeseries_imag", T.types[1], ((obs.elsize...,obs.count),(obs.elsize...,-1)), "chunk", (obs.elsize...,obs.alloc), "compress", 9)
      end
      new_count = obs.count
    else
      # update counter
      new_count = read(f["obs/" * obs.name * "/count"]) + obs.count
      o_delete(f, "obs/" * obs.name * "/count")
      write(f, "obs/" * obs.name * "/count", new_count)
    end
    g = f["obs/" * obs.name]

    # update (append to) time series
    colons = [Colon() for k in 1:obs.eldims]

    if T<:Real
      set_dims!(g["timeseries"],(obs.elsize...,new_count))
      g["timeseries"][colons...,end-obs.count+1:end] = obs.timeseries[colons...,:]

      # if obs.eldims == 0
      o_delete(g, "mean")
      m = sum(read(g["timeseries"]), ndims(obs.timeseries))/new_count
      write(g, "mean", m)
      # end
    else
      set_dims!(g["timeseries_real"],(obs.elsize...,new_count))
      set_dims!(g["timeseries_imag"],(obs.elsize...,new_count))
      g["timeseries_real"][colons...,end-obs.count+1:end] = real(obs.timeseries[colons...,:])
      g["timeseries_imag"][colons...,end-obs.count+1:end] = imag(obs.timeseries[colons...,:])
    end

  end
end


function confs2hdf5{T}(filename::String, obs::Observable{T})

  if obs.count != obs.alloc
    warn("Saving incomplete time series chunk of observable \"$obs.name\".")
  end

  h5open(filename, isfile(filename)?"r+":"w") do f

    new_count = -1
    if !exists(f, "configurations")
      # initialze chunk storage
      write(f, "count", obs.count)
      d_create(f, "configurations", T, ((obs.elsize...,obs.count),(obs.elsize...,-1)), "chunk", (obs.elsize...,obs.alloc), "compress", 9)
      new_count = obs.count
    else
      # update counter
      new_count = read(f["count"]) + obs.count
      o_delete(f, "count")
      write(f, "count", new_count)
    end

    # update (append to) time series
    colons = [Colon() for k in 1:obs.eldims]

    set_dims!(f["configurations"],(obs.elsize...,new_count))
    f["configurations"][colons...,end-obs.count+1:end] = obs.timeseries[colons...,:]

  end
end


function hdf52obs(filename::String, obsname::String)
  h5open(filename, "r+") do f
    if !exists(f, "/obs/" * obsname) error("Observable does not exist in \"$filename\"")
    else
      o = f["/obs/" * obsname]
      if exists(o, "timeseries") # Real
        chunksize = get_chunk(o["timeseries"])[end]
        elsize = size(o["timeseries"])[1:end-1]
        obs = Observable{eltype(o["timeseries"])}(obsname,elsize,chunksize)
        obs.timeseries = read(o["timeseries"])
        obs.count = size(obs.timeseries)[end]
        return obs
      else # Complex
        chunksize = get_chunk(o["timeseries_real"])[end]
        elsize = size(o["timeseries_real"])[1:end-1]
        obs = Observable{Complex{eltype(o["timeseries_real"])}}(obsname,elsize,chunksize)
        obs.timeseries = read(o["timeseries_real"]) + im*read(o["timeseries_imag"])
        obs.count = size(obs.timeseries)[end]
        return obs
      end
    end
  end
end


function clear(obs::Observable)
  obs.count = 0
end

function isempty(obs::Observable)
  return (obs.count == 0)
end

function delete(filename::String, obsname::String)
  h5open(filename, "r+") do f
    if !exists(f, "obs/" * obsname)
      info("Nothing to be done.")
    else
      o_delete(f, "obs/" * obsname)
    end
  end
end
delete(filename::String, obs::Observable) = delete(filename, obs.name)


function listobs(filename::String)
  h5open(filename, "r+") do f
    return names(f["/obs"])
  end
end
