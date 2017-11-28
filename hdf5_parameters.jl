using HDF5

"""
    parameters2hdf5(p::Parameters, filename)
    
Save `p` to HDF5 file (e.g. `.out.h5`).
"""
function parameters2hdf5(p::Parameters, filename::String)
  isfile(filename)?f = h5open(filename, "r+"):f = h5open(filename, "w")
  for i in 1:nfields(Parameters)
    field = fieldname(Parameters, i)
    field_name = string(field)
    field_value = getfield(p, field)
    field_type = typeof(field_value)

    if field_type == Distributions.Uniform{Float64}
      field_value = field_value.b
    elseif field_type == Bool
      # HDF5 1.8.x doesn't support Bool fields, use Int instead.
      field_value = Int(field_value)
    end

    try
      if HDF5.exists(f, "params/" * field_name)
        if read(f["params/"*field_name]) != field_value
          close(f)
          error(field_name, " exists but differs from current ")
        end
      else
        f["params/" * field_name] = field_value
      end
    catch e
      close(f)
      warn("Error in dumping parameters to hdf5: ", e)
    end
  end
  close(f)
end

"""
    hdf52parameters!(p::Parameters, input_h5)
    
Load `p` from HDF5 file (e.g. `.out.h5`).
"""
function hdf52parameters!(p::Parameters, input_h5::String)
  fields = fieldnames(Parameters)
  HDF5.h5open(input_h5, "r") do f
    try
      for field_name in names(f["params"])
        field = Symbol(field_name)
        if field in fields

          value = read(f["params/$(field_name)"])
          if field_name in ["global_updates", "chkr", "Bfield", "all_checks"] # handle Bools
            value = Bool(value)
          elseif field_name in ["box", "box_global"] # handle Distributions
            value = Distributions.Uniform{Float64}(-value, value)
          end

          setfield!(p, field, value)

        else
          warn("HDF5 contains \"params/$(field_name)\" which is not a field of type Parameters! Maybe old file? Try `hdf52parameters!_old`.")
        end
      end
    catch e
      error("Error in loading Parameters object from HDF5: ", e)
    end
  end

  deduce_remaining_parameters(p)

  nothing
end


"""
    hdf52parameters!_old(p::Parameters, filename)
    
Deprecated version of `hdf52parameters!`. Use it only for old data (created before 28.11.2017).
"""
function hdf52parameters!_old(p::Parameters, input_h5::String)
  warn("DEPRECATED: Only use for old data (where we dumped `params::Dict`). Should now use `hdf52parameters!` instead.")
  # READ Parameters from h5 file
  if input_h5[end-2:end] == "jld"
    f = jldopen(input_h5, "r")
  else
    f = HDF5.h5open(input_h5, "r")
  end

  params = Dict{Any, Any}()

  try
    for e in names(f["params"])
           params[e] = read(f["params/$e"])
    end
  catch e
    println(e)
  end

  set_parameters(p, params)
  close(f)

  return params
end