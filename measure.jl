# -------------------------------------------------------
#            Check input arguments
# -------------------------------------------------------
# measure.jl input args: runstable.jld [inputfile.meas.xml]
#
# If no meas.xml given (second argument is zero), we only measure standard bosonic stuff.
length(ARGS) < 1 && error("input args: runstable.jld [inputfile.meas.xml]")



# -------------------------------------------------------
#            Start + Check git branch
# -------------------------------------------------------
const start_time = now()
println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))
println("Hostname: ", gethostname())

using Git
branch = Git.branch(dir=dirname(@__FILE__)).string[1:end-1]
if branch != "master"
  println("!!!Not on branch master but \"$(branch)\"!!!")
  flush(STDOUT)
end




# #######################################################
#                       Framework
# #######################################################
# -------------------------------------------------------
#                       Includes
# -------------------------------------------------------
include("dqmc_framework.jl")
using Parameters, JLD, DataFrames





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

  safe_mult::Int = 10
  overwrite::Bool = false
  num_threads::Int = 1
  include_running::Bool = true
  walltimelimit::Dates.DateTime = choose_walltimelimit(ENV)



  # temporary variables (change per run)
  dochi_dyn::Bool = chi_dyn
  dochi_dyn_symm::Bool = chi_dyn_symm
  dobinder::Bool = binder
  doetpc::Bool = etpc
  outfile::String = ""
end

donothing!(mp::MeasParams) = begin
  for f in dofields
    setfield!(mp, f, false)
  end
end
doallrequested!(mp::MeasParams) = begin
  @inbounds for (i,f) in enumerate(dofields)
    setfield!(mp, f, dodefaults[i])
  end
end




# -------------------------------------------------------
#              Initialize stack for meas
# -------------------------------------------------------
function initialize_stack(mc::AbstractDQMC, mp::MeasParams)
  @mytimeit mc.a.to "initialize_stack" begin
    _initialize_stack(mc)

    # allocate for measurement run based on todos
    mp.doetpc && allocate_etpc!(mc)

  end #timeit

  nothing
end





# -------------------------------------------------------
#                   For each run ...
# -------------------------------------------------------
function foreachrun(mp::MeasParams, rt::DataFrame)
  for r in eachrow(rt) # for every selected run
    inxml = r[:PATH] # path to file
    cd(dirname(inxml))

    # --------------- Find .out.h5 ----------------------
    fpath = replace(inxml, ".in.xml", ".out.h5")
    if !isfile(fpath)
        fpath = replace(fpath, ".out.h5", ".out.h5.running")
        isfile(fpath) || continue
    end
    fpathnice = replace(fpath, "/projects/ag-trebst/bauer/", "")
    println(fpathnice); flush(STDOUT)


    # ----------- Set .meas.h5 and todos ------------------
    mp.outfile = replace(fpath, ".out.h5", ".meas.h5")
    doallrequested!(mp)
    if !endswith(mp.outfile, ".running")
        # already measured
        if !mp.overwrite
          isfile(mp.outfile) && begin
            println("Measurements found."); # only measure what isn't there yet
            donothing!(mp)

            todo = false
            println("Checking...")
            allobs = listobs(mp.outfile)

            mp.chi_dyn && !("chi_dyn" in allobs) && (mp.dochi_dyn = true; todo = true)
            mp.chi_dyn_symm && !("chi_dyn_symm" in allobs) && (mp.dochi_dyn_symm = true; todo = true)
            mp.binder && !("binder" in allobs) && (mp.dobinder = true; todo = true)
            mp.etpc && !("etpc_minus" in allobs) && (mp.doetpc = true; todo = true)
            mp.etpc && !("etpc_plus" in allobs) && (mp.doetpc = true; todo = true)

            todo || begin println("Already measured. Skipping."); continue end
          end
        end # measure everything requested (overwrite if necessary)

        # maybe job finished in the mean time. is there an old .meas.h5.running? if so, delete it.
        isfile(mp.outfile*".running") && rm(mp.outfile*".running")
    end




    # -------------- Load DQMC params/results -----------------
    local confs;
    local p;
    try
        # load configurations (TODO: what to actually load here?)
        confs = ts_flat(fpath, "obs/configurations")

        # load dqmc params
        p = Params(); xml2parameters!(p, inxml, false);
        # TODO: Probably even set up mc object here?
    catch err
        println("Failed to load dqmc params/results for $(fpathnice).")
        println("Maybe it doesn't have any configurations yet?")
        println("Error: $(err)")
        println()
    end

    # --------------------- Measure -------------------------
    measure(mp, p, confs)

    println("Done.\n")
    flush(STDOUT)

    if now() >= mp.walltimelimit
      println("Approaching wall-time limit. Safely exiting.")
      exit(42)
    end
  end
end













# -------------------------------------------------------
#                      MEASUREMENTS
# -------------------------------------------------------
function measure(mp::MeasParams, p::Params, confs::AbstractArray{Float64, 4})
  const num_confs = size(confs, ndims(confs))
  const nsweeps = num_confs * p.write_every_nth

  # ------------------- Allocate --------------------------
  # chi_dyn
  mp.dochi_dyn_symm && (chi_dyn_symm = Observable(Array{Float64, 3}, "chi_dyn_symm"; alloc=num_confs))
  mp.dochi_dyn && (chi_dyn = Observable(Array{Float64, 3}, "chi_dyn"; alloc=num_confs))
  
  # binder
  m2s = Vector{Float64}(num_confs)
  m4s = Vector{Float64}(num_confs)



  # ----------------- Measure loop ------------------------
  mp.dochi_dyn && println("Measuring chi_dyn/chi_dyn_symm/binder etc. ...");
  flush(STDOUT)

  @inbounds @views for i in 1:num_confs

      # chi_dyn
      if mp.dochi_dyn
        # chi_dyn
        chi = measure_chi_dynamic(confs[:,:,:,i])
        add!(chi_dyn, chi)

        if mp.dochi_dyn_symm
          chi = (permutedims(chi, [2,1,3]) + chi)/2 # C4 is basically flipping qx and qy (which only go from 0 to pi since we perform a real fft.)
          add!(chi_dyn_symm, chi)
        end
      end

      # binder
      if mp.dobinder
        # binder
        m = mean(confs[:,:,:,i],[2,3])
        m2s[i] = dot(m, m)
        m4s[i] = m2s[i]*m2s[i]
      end
  end



  # ------------ Postprocessing   ----------------
  if mp.dobinder
    # binder postprocessing
    m2ev2 = mean(m2s)^2
    m4ev = mean(m4s)

    binder = Observable(Float64, "binder")
    add!(binder, m4ev/m2ev2)
  end

  

  # ------------ Export results   ----------------
  println("Calculating errors and exporting..."); flush(STDOUT)
  mp.dochi_dyn && export_result(chi_dyn, mp.outfile, "obs/chi_dyn"; timeseries=true)
  mp.dochi_dyn_symm && export_result(chi_dyn_symm, mp.outfile, "obs/chi_dyn_symm"; timeseries=true)

  mp.dobinder && export_result(binder, mp.outfile, "obs/binder", error=false) # jackknife for error

  h5open(mp.outfile, "r+") do fout
    HDF5.has(fout, "nsweeps") && HDF5.o_delete(fout, "nsweeps")
    HDF5.has(fout, "write_every_nth") && HDF5.o_delete(fout, "write_every_nth")
    fout["nsweeps"] = nsweeps
    fout["write_every_nth"] = p.write_every_nth
  end
end







# #######################################################
#                       Main
# #######################################################
# -------------------------------------------------------
#           Parse meas.xml and runstable
# -------------------------------------------------------
input_xml = length(ARGS) == 2 ? ARGS[2] : ""


# meas.xml: direct mapping of xml fields to kwargs
if input_xml != ""
  mpdict = xml2dict(input_xml, false)
  kwargs = Dict([Symbol(lowercase(k))=>lowercase(v) for (k,v) in mpdict])
  mp = MeasParams(; kwargs...)
else
  mp = MeasParams()
end

const dofields = Symbol.(filter(x->startswith(x, "do"), string.(fieldnames(MeasParams))))
const dodefaults = getfield.(mp, Symbol.(replace.(string.(dofields), "do", "")))

# runstable
try
  global rt = load(ARGS[1], "runstable")
catch
  error("Couldn't load runstable.")
end


# set num threads
try
  BLAS.set_num_threads(mp.num_threads)
  FFTW.set_num_threads(1)
  ENV["OMP_NUM_THREADS"] = mp.num_threads
  ENV["MKL_NUM_THREADS"] = mp.num_threads
  ENV["JULIA_NUM_THREADS"] = mp.num_threads
end


# -------------------------------------------------------
#                     Let's go
# -------------------------------------------------------
pwd_before = pwd()
foreachrun(mp, rt)
cd(pwd_before)


# -------------------------------------------------------
#                       Exit
# -------------------------------------------------------
end_time = now()
println("\nEnded: ", Dates.format(end_time, "d.u yyyy HH:MM"))
@printf("Duration: %.2f minutes", (end_time - start_time).value/1000./60.)