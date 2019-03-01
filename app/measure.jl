# -------------------------------------------------------
#            Check input arguments
# -------------------------------------------------------
length(ARGS) >= 1 || error("input arg: prefix.meas.xml [or prefix.in.xml]")

# support for symbol link to app/measure.jl in root of repo
__FILE__ = basename(pwd()) == "app" ? "" : joinpath(dirname(@__FILE__), "app")

# -------------------------------------------------------
#            Start + Check git branch
# -------------------------------------------------------
using Dates, Printf, Pkg
const start_time = now()
println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))
println("Hostname: ", gethostname())



function is_dqmc_env_activated()
    project_file = Base.active_project()
    project = Pkg.Types.read_project(project_file)
    return !isnothing(project.name) && lowercase(project.name) == "dqmc"
end

if !is_dqmc_env_activated()
  println("Activating DQMC environment.")
  haskey(ENV, "JULIA_DQMC") || error("DQMC environment not loaded and JULIA_DQMC env variable not set!")
  Pkg.activate(ENV["JULIA_DQMC"])
  is_dqmc_env_activated() || error("Invalid JULIA_DQMC env variable value.")
end



using Git
branch = Git.branch(dir=dirname(@__FILE__)).string[1:end-1]
if branch != "master"
  println("!!!Not on branch master but \"$(branch)\"!!!")
  flush(stdout)
end




# #######################################################
#                       Framework
# #######################################################
# -------------------------------------------------------
#                       Includes
# -------------------------------------------------------
include(joinpath(__FILE__), "../src/dqmc_framework.jl"))
using Parameters
using ProgressMeter
using RecursiveArrayTools




# -------------------------------------------------------
#                  Define MeasParams
# -------------------------------------------------------
@with_kw mutable struct MeasParams
  # bosonic
  chi_dyn::Bool = true
  chi_dyn_symm::Bool = true
  binder::Bool = true

  # fermionic
  etpc::Bool = false
  tdgfs::Bool = false

  safe_mult::Int = 10
  overwrite::Bool = false
  num_threads::Int = 1
  include_running::Bool = true

  outfile::String = ""
  inxml::String = ""
  dqmc_outfile::String = ""
  chunksize::Int = 100
end



function measxml2MeasParams(fname)
  # meas.xml: direct mapping of xml fields to kwargs
  mpdict = xml2dict(fname, false)
  kwargs_strings = Dict([Symbol(lowercase(k))=>lowercase(v) for (k,v) in mpdict])

  kwargs = Dict()
  for (k, v) in kwargs_strings
    if v in ["true", "false"]
      kwargs[k] = parse(Bool, v)
    else
      try
        kwargs[k] = parse(Int64, v)
      catch
        kwargs[k] = v
      end
    end
  end

  mp = MeasParams(; kwargs...)
  mp.inxml = fname[1:end-9]*".in.xml"
  return mp
end



"""
Returns `true` if we want to measure fermionic stuff.
"""
@inline hasfermionic(mp::MeasParams) = mp.etpc || mp.tdgfs
@inline needs_mc(mp::MeasParams) = mp.etpc || mp.tdgfs
@inline needs_etgf(mp::MeasParams) = mp.etpc

# see https://discourse.julialang.org/t/is-the-first-key-of-a-namedtuple-special/21256
@inline hasfermionic(ol::NamedTuple{K,V}) where {K,V} = Base.sym_in(:etpc_plus, K) ||
                                                        Base.sym_in(:tdgfs_Gt0, K)
@inline needs_etgf(ol::NamedTuple{K,V}) where {K,V} = Base.sym_in(:etpc_plus, K)



# -------------------------------------------------------
#              Initialize stack for meas
# -------------------------------------------------------
function initialize_stack_for_measurements(mc::AbstractDQMC, mp::MeasParams)
    _initialize_stack(mc)

    # allocate for measurement run based on todos
    mp.etpc && allocate_etpc!(mc)
    mp.tdgfs && allocate_tdgfs!(mc)

    nothing
end





# -------------------------------------------------------
#                        Main
# -------------------------------------------------------
function main(mp::MeasParams)
  # --------------- Find .out.h5 ----------------------
  mp.dqmc_outfile = replace(mp.inxml, ".in.xml" => ".out.h5")
  if !isfile(mp.dqmc_outfile)
      mp.dqmc_outfile = replace(mp.dqmc_outfile, ".out.h5" => ".out.h5.running")
      isfile(mp.dqmc_outfile) || error("Couldn't find .out.h5[.running] file.")
  end


  # ----------- Set .meas.h5 and todos ------------------
  mp.outfile = replace(mp.dqmc_outfile, ".out.h5" => ".meas.h5")
  if !endswith(mp.outfile, ".running")
      # already measured
      if !mp.overwrite
        isfile(mp.outfile) && begin
          println("Measurements found."); # only measure what isn't there yet

          todo = false
          println("Checking...")
          allobs = listobs(mp.outfile)

          mp.chi_dyn && !("chi_dyn" in allobs) && (todo = true)
          mp.chi_dyn_symm && !("chi_dyn_symm" in allobs) && (todo = true)
          mp.binder && !("binder" in allobs) && (todo = true)
          mp.etpc && !("etpc_minus" in allobs) && (todo = true)
          mp.etpc && !("etpc_plus" in allobs) && (todo = true)
          mp.tdgfs && !("tdgfs_Gt0" in allobs) && (todo = true)
          mp.tdgfs && !("tdgfs_G0t" in allobs) && (todo = true)

          todo || begin println("Already measured."); return;  end
        end
      else
        # measure everything requested (overwrite if necessary)
        println("\nOVERWRITE MODE")
      end

      # maybe job finished in the mean time. is there an old .meas.h5.running? if so, delete it.
      isfile(mp.outfile*".running") && rm(mp.outfile*".running")
  end




  # -------------- Load DQMC params/results -----------------
  println("\nLoading DQMC results...")

  greens = nothing; 
  mc = nothing;
  local confs
  local chunkcount
  try
    confs = ts_flat(mp.dqmc_outfile, "obs/configurations")
    chunkcount = h5read(mp.dqmc_outfile, "obs/configurations/timeseries/chunk_count")
    mp.chunksize = h5read(mp.dqmc_outfile, "obs/configurations/alloc")
  catch err
    println("Couldn't read configuration data. Probably no configurations yet? Exiting.")
    exit()
  end

  needs_etgf(mp) && (greens = loadobs_frommemory(mp.dqmc_outfile, "obs/greens"))

  # load dqmc params
  p = Params(); xml2parameters!(p, mp.inxml, false);


  # ------------------- Create Observables --------------------
  num_confs = size(confs, ndims(confs))
  nsweeps = num_confs * p.write_every_nth
  obs = NamedTuple() # list of Observables

  # chi_dyn
  mp.chi_dyn_symm && (obs = add(obs, chi_dyn_symm = Observable(Array{Float64, 3}, "chi_dyn_symm"; alloc=num_confs)))
  mp.chi_dyn && (obs = add(obs, chi_dyn = Observable(Array{Float64, 3}, "chi_dyn"; alloc=num_confs)))
  
  # binder
  mp.binder && (obs = add(obs, m2s = Vector{Float64}(undef, num_confs)))
  mp.binder && (obs = add(obs, m4s = Vector{Float64}(undef, num_confs)))
  mp.binder && (obs = add(obs, binder = Observable(Float64, "binder")))


  # prepare fermionic sector if necessary
  if needs_etgf(mp)
    mc = DQMC(p)
    initialize_stack_for_measurements(mc, mp)
  end


  # etpc
  if mp.etpc
    (obs = add(obs, etpc_plus = Observable(Matrix{Float64}, "S-wave equal time pairing susceptibiliy (ETPC plus)", alloc=num_confs)))
    (obs = add(obs, etpc_minus = Observable(Matrix{Float64}, "D-wave equal time pairing susceptibiliy (ETPC minus)", alloc=num_confs)))
  end

  # tdgfs
  if mp.tdgfs
    (obs = add(obs, tdgfs_Gt0 = Observable(Array{geltype(mc), 3}, "Time-displaced Green's function G(tau,0) (TDGF Gt0)", alloc=num_confs)))
    (obs = add(obs, tdgfs_G0t = Observable(Array{geltype(mc), 3}, "Time-displaced Green's function G(0, tau) (TDGF G0t)", alloc=num_confs)))
  end

  # --------------------- Measure -------------------------
  print("\nPrepared Observables ");
  println(keys(obs))
  println()
  measure(mp, p, obs, confs, greens, mc)

  # ------------ Export results   ----------------
  export_results(mp, p, obs, nsweeps)
end






















# -------------------------------------------------------
#                      MEASUREMENTS
# -------------------------------------------------------
function measure(mp::MeasParams, p::Params, obs::NamedTuple{K,V}, confs, greens, mc) where {K,V}
    # greens = (lazy) observable of greens functions or nothing
    num_confs = size(confs, ndims(confs))
    nsweeps = num_confs * p.write_every_nth

    # ----------------- Measure loop ------------------------
    length(obs) > 0 && println("Measuring ...");
    flush(stdout)

    @inbounds @views @showprogress for i in 1:num_confs
        conf = confs[:,:,:,i]

        measure_bosonic(mp, p, obs, conf, i)

        # ifs should all happen at compile time (based on obs, i.e. NamedTuple keys)
        if hasfermionic(obs)
            g = needs_etgf(obs) ? greens[i] : nothing # load single greens from disk per MCO.jl
            measure_fermionic(mp, p, obs, conf, g, mc, i)
        end
    end


    # ------------ Postprocessing   ----------------
    if :binder in keys(obs)
        # binder postprocessing
        m2ev2 = mean(obs[:m2s])^2
        m4ev = mean(obs[:m4s])

        add!(obs[:binder], m4ev/m2ev2)
    end
end







function measure_bosonic(mp, p, obs, conf, i)
      # chi_dyn
      if :chi_dyn in keys(obs)
        chi = measure_chi_dynamic(conf)
        add!(obs[:chi_dyn], chi)

        if :chi_dyn_symm in keys(obs)
          chi = (permutedims(chi, [2,1,3]) + chi)/2 # C4 is basically flipping qx and qy (which only go from 0 to pi since we perform a real fft.)
          add!(obs[:chi_dyn_symm], chi)
        end
      end

      # binder
      if :binder in keys(obs)
        m = mean(conf, dims=(2,3))
        obs[:m2s][i] = dot(m, m)
        obs[:m4s][i] = obs[:m2s][i] * obs[:m2s][i]
      end
      nothing
end





function measure_fermionic(mp, p, obs, conf, greens, mc, i)
    # etpc
    if :etpc_plus in keys(obs)
        mc.p.hsfield = conf
        etpc!(mc, greens)
        add!(obs[:etpc_plus], mc.s.meas.etpc_plus)
        add!(obs[:etpc_minus], mc.s.meas.etpc_minus)
    end


    # tdgfs
    if :tdgfs_Gt0 in keys(obs)
        mc.p.hsfield = conf
        calc_tdgfs!(mc)
        add!(obs[:tdgfs_Gt0], VectorOfArray(mc.s.meas.Gt0))
        add!(obs[:tdgfs_G0t], VectorOfArray(mc.s.meas.G0t))
    end
    nothing
end









function export_results(mp, p, obs, nsweeps)
    println("Calculating errors and exporting..."); flush(stdout)
    :chi_dyn in keys(obs) && export_result(obs[:chi_dyn], mp.outfile, "obs/chi_dyn"; timeseries=true)
    :chi_dyn_symm in keys(obs) && export_result(obs[:chi_dyn_symm], mp.outfile, "obs/chi_dyn_symm"; timeseries=true)

    :binder in keys(obs) && export_result(obs[:binder], mp.outfile, "obs/binder", error=false) # jackknife for error

    :etpc_plus in keys(obs) && export_result(obs[:etpc_plus], mp.outfile, "obs/etpc_plus"; timeseries=true)
    :etpc_minus in keys(obs) && export_result(obs[:etpc_minus], mp.outfile, "obs/etpc_minus"; timeseries=true)

    :tdgfs_Gt0 in keys(obs) && export_result(obs[:tdgfs_Gt0], mp.outfile, "obs/tdgfs_Gt0"; timeseries=true)
    :tdgfs_G0t in keys(obs) && export_result(obs[:tdgfs_G0t], mp.outfile, "obs/tdgfs_G0t"; timeseries=true)

    h5open(mp.outfile, "r+") do fout
        HDF5.has(fout, "nsweeps") && HDF5.o_delete(fout, "nsweeps")
        HDF5.has(fout, "write_every_nth") && HDF5.o_delete(fout, "write_every_nth")
        fout["nsweeps"] = nsweeps
        fout["write_every_nth"] = p.write_every_nth
    end
    println("Done. Exported to $(mp.outfile).")
    flush(stdout)
end












# -------------------------------------------------------
#                 Helpers and stuff
# -------------------------------------------------------

"""
Allows one to use keyword syntax to add new entries to a `NamedTuple` `obs`.
"""
add(obs::NamedTuple; kw...) = combine(obs, values(kw))
"""
Combine two `NamedTuple`s into a new one.
"""
combine(x::NamedTuple, y::NamedTuple) = NamedTuple{(keys(x)..., keys(y)...)}((x...,y...))






# #######################################################
#                    Start script
# #######################################################
# -------------------------------------------------------
#           Parse ARGS and maybe .meas.xml
# -------------------------------------------------------

const arg = ARGS[1]

isfile(arg) || error("Input file $arg not found.")
if endswith(arg, ".meas.xml")
  mp = measxml2MeasParams(arg)
elseif endswith(arg, ".in.xml")
  measxml = arg[1:end-7]*".meas.xml"
  isfile(measxml) || error("File $(measxml) not found. Input arg was: $(arg).")
  mp = measxml2MeasParams(measxml)
else
  error("No .meas.xml or .in.xml file found. Input arg was: $(arg)")
end

isfile(mp.inxml) || error("DQMC input file $(mp.inxml) not found.")



# set num threads
try
  BLAS.set_num_threads(mp.num_threads)
  FFTW.set_num_threads(mp.num_threads)
  ENV["OMP_NUM_THREADS"] = mp.num_threads
  ENV["MKL_NUM_THREADS"] = mp.num_threads
  ENV["JULIA_NUM_THREADS"] = 1
catch
end


# -------------------------------------------------------
#                     Let's go
# -------------------------------------------------------
main(mp)


# -------------------------------------------------------
#                       Exit
# -------------------------------------------------------
end_time = now()
println("\nEnded: ", Dates.format(end_time, "d.u yyyy HH:MM"))
@printf("Duration: %.2f minutes", (end_time - start_time).value / 1000 / 60)