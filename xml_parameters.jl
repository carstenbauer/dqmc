# conversion between parameters::Dict (NOT ::Parameters !!), xml file and hdf5 file
using LightXML
using Iterators
import HDF5

function xml2parameters(fname::String)
  params = Dict{Any, Any}()
  xdoc = parse_file(fname)
  xroot = LightXML.root(xdoc)
  for c in child_nodes(xroot)
    if is_elementnode(c)
      for p in child_nodes(c)
        if is_elementnode(p)
          e = XMLElement(p)
          if LightXML.name(e) == "PARAMETER"
            println(attribute(e, "name"), " = ", content(e))
            params[attribute(e, "name")] = content(e)
          end
        end
      end
    end
  end
  return params
end

function parameters2hdf5(params::Dict, filename::String)
  f = HDF5.h5open(filename, "r+")
  for (k, v) in params
    try
      if HDF5.exists(f, "params/" * k)
        if read(f["params/"*k]) != v
          error(k, " exists but differs from current ")
        end
      else
        f["params/" * k] = v
      end
    catch e
    end

  end

  close(f)
end

function parameterset2xml(params::Dict, prefix::String)
  for (i, param_set) in enumerate(product(values(params)...))
    xdoc = XMLDocument()
    xroot = create_root(xdoc, "SIMULATION")
    pn = new_child(xroot, "PARAMETERS")
    for (k, p) in zip(keys(params), param_set)
      pc = new_child(pn, "PARAMETER")
      add_text(pc, string(p))
      set_attribute(pc, "name", string(k))
    end
    save_file(xdoc, prefix * ".task" * string(i) * ".in.xml")
  end
end
