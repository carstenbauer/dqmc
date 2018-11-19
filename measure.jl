# -------------------------------------------------------
#            Check input arguments
# -------------------------------------------------------
length(ARGS) == 1 || error("input arg: prefix.meas.xml [or just prefix or prefix.in.xml or prefix.out.h5 (will use default settings)]")



# -------------------------------------------------------
#            Start + Check git branch
# -------------------------------------------------------
using Dates, Printf
const start_time = now()
println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))
println("Hostname: ", gethostname())

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
include("dqmc_framework.jl")
using Parameters
using ProgressMeter




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

  outfile::String = ""
  inxml::String = ""
end



# -------------------------------------------------------
#              Initialize stack for meas
# -------------------------------------------------------
function initialize_stack(mc::AbstractDQMC, mp::MeasParams)
  @mytimeit mc.a.to "initialize_stack" begin
    _initialize_stack(mc)

    # allocate for measurement run based on todos
    mp.etpc && allocate_etpc!(mc)

  end #timeit

  nothing
end





# -------------------------------------------------------
#                        Main
# -------------------------------------------------------
function main(mp::MeasParams)
  # --------------- Find .out.h5 ----------------------
  h5path = replace(mp.inxml, ".in.xml" => ".out.h5")
  if !isfile(h5path)
      h5path = replace(h5path, ".out.h5" => ".out.h5.running")
      isfile(h5path) || error("Couldn't find .out.h5[.running] file.")
  end


  # ----------- Set .meas.h5 and todos ------------------
  mp.outfile = replace(h5path, ".out.h5" => ".meas.h5")
  if !endswith(mp.outfile, ".running")
      # already measured
      if !mp.overwrite
        isfile(mp.outfile) && begin
          println("Measurements found."); # only measure what isn't there yet

          todo = false
          println("Checking...")
          allobs = listobs(mp.outfile)

          mp.chi_dyn && !("chi_dyn" in allobs) && (mp.chi_dyn = true; todo = true)
          mp.chi_dyn_symm && !("chi_dyn_symm" in allobs) && (mp.chi_dyn_symm = true; todo = true)
          mp.binder && !("binder" in allobs) && (mp.binder = true; todo = true)
          mp.etpc && !("etpc_minus" in allobs) && (mp.etpc = true; todo = true)
          mp.etpc && !("etpc_plus" in allobs) && (mp.etpc = true; todo = true)

          todo || begin println("Already measured."); return;  end
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
      confs = ts_flat(h5path, "obs/configurations")

      # load dqmc params
      p = Params(); xml2parameters!(p, mp.inxml, false);
  catch err
      println("Failed to load dqmc params/results for $(h5path).")
      println("Maybe it doesn't have any configurations yet?")
      error(err)
  end

  # --------------------- Measure -------------------------
  measure(mp, p, confs)

  println("Done.\n")
  flush(stdout)
end










# -------------------------------------------------------
#                      MEASUREMENTS
# -------------------------------------------------------
function measure(mp::MeasParams, p::Params, confs::AbstractArray{Float64, 4})
  num_confs = size(confs, ndims(confs))
  nsweeps = num_confs * p.write_every_nth

  # ------------------- Allocate --------------------------
  # chi_dyn
  mp.chi_dyn_symm && (chi_dyn_symm = Observable(Array{Float64, 3}, "chi_dyn_symm"; alloc=num_confs))
  mp.chi_dyn && (chi_dyn = Observable(Array{Float64, 3}, "chi_dyn"; alloc=num_confs))
  
  # binder
  mp.binder && (m2s = Vector{Float64}(undef, num_confs))
  mp.binder && (m4s = Vector{Float64}(undef, num_confs))

  # etpc
  if mp.etpc
    mc = DQMC(p)
    initialize_stack(mc, mp)
    Pplus = Observable(Matrix{Float64}, "S-wave equal time pairing susceptibiliy (ETPC plus)", alloc=num_confs)
    Pminus = Observable(Matrix{Float64}, "D-wave equal time pairing susceptibiliy (ETPC minus)", alloc=num_confs)
  end


  # ----------------- Measure loop ------------------------
  mp.chi_dyn && println("Measuring ...");
  flush(stdout)

  @inbounds @views @showprogress for i in 1:num_confs

      # chi_dyn
      if mp.chi_dyn
        chi = measure_chi_dynamic(confs[:,:,:,i])
        add!(chi_dyn, chi)

        if mp.chi_dyn_symm
          chi = (permutedims(chi, [2,1,3]) + chi)/2 # C4 is basically flipping qx and qy (which only go from 0 to pi since we perform a real fft.)
          add!(chi_dyn_symm, chi)
        end
      end

      # binder
      if mp.binder
        m = mean(confs[:,:,:,i], dims=(2,3))
        m2s[i] = dot(m, m)
        m4s[i] = m2s[i]*m2s[i]
      end

      # etpc
      if mp.etpc
        mc.p.hsfield = confs[:,:,:,i]
        etpc!(mc, measure_greens(mc))
        add!(Pplus, mc.s.meas.etpc_plus)
        add!(Pminus, mc.s.meas.etpc_minus)
      end
  end



  # ------------ Postprocessing   ----------------
  if mp.binder
    # binder postprocessing
    m2ev2 = mean(m2s)^2
    m4ev = mean(m4s)

    binder = Observable(Float64, "binder")
    add!(binder, m4ev/m2ev2)
  end

  

  # ------------ Export results   ----------------
  println("Calculating errors and exporting..."); flush(stdout)
  mp.chi_dyn && export_result(chi_dyn, mp.outfile, "obs/chi_dyn"; timeseries=true)
  mp.chi_dyn_symm && export_result(chi_dyn_symm, mp.outfile, "obs/chi_dyn_symm"; timeseries=true)

  mp.binder && export_result(binder, mp.outfile, "obs/binder", error=false) # jackknife for error

  mp.etpc && export_result(Pplus, mp.outfile, "obs/Pplus"; timeseries=true)
  mp.etpc && export_result(Pminus, mp.outfile, "obs/Pminus"; timeseries=true)

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
#           Parse ARGS and maybe .meas.xml
# -------------------------------------------------------

const arg = ARGS[1]

if endswith(arg, ".meas.xml")
  isfile(arg) || error("Meas input file $arg not found.")
  # meas.xml: direct mapping of xml fields to kwargs
  mpdict = xml2dict(arg, false)
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
  mp.inxml = arg[1:end-9]*".in.xml"
else
  mp = MeasParams()

  if endswith(arg, ".in.xml")
    mp.inxml = arg
  else
    # assume arg to be a simple prefix
    mp.inxml = arg*".in.xml"
  end
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