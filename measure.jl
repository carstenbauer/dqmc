# measure.jl input args: runstable.jld [inputfile.meas.xml]
#
# If no meas.xml given (second argument is zero), we only measure standard bosonic stuff.
length(ARGS) < 1 && error("input args: runstable.jld [inputfile.meas.xml]")

# -------------------------------------------------------
#        Start + wall-time handling
# -------------------------------------------------------
start_time = now()
println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))
println("Hostname: ", gethostname())

# Calculate DateTime where wall-time limit will be reached.
function wtl2DateTime(wts::AbstractString, start_time::DateTime)
  @assert contains(wts, "-")
  @assert contains(wts, ":")
  @assert length(wts) >= 10

  tmp = split(wts, "-")

  d = parse(Int, tmp[1])
  h, m, s = parse.(Int, split(tmp[2], ":"))

  start_time + Dates.Day(d) + Dates.Hour(h) + Dates.Minute(m) + Dates.Second(s)
end


# -------------------------------------------------------
#            Includes
# -------------------------------------------------------
include("dqmc_framework.jl")
using JLD, DataFrames


# -------------------------------------------------------
#    Parse meas.xml, runstable and create MeasParams
# -------------------------------------------------------
input_xml = length(ARGS) == 2 ? ARGS[2] : ""

using Git
branch = Git.branch(dir=dirname(@__FILE__)).string[1:end-1]
if branch != "master"
  println("!!!Not on branch master but \"$(branch)\"!!!")
  flush(STDOUT)
end

using Parameters
@with_kw mutable struct MeasParams
  chi::Bool = true
  chi_symm::Bool = true
  binder::Bool = true
  greens::Bool = false

  safe_mult::Int = 10
  overwrite::Bool = false
  num_threads::Int = 1
  include_running::Bool = true
  walltimelimit::Dates.DateTime = Dates.DateTime("2099", "YYYY") # effective infinity
end

# direct mapping of xml fields to kwargs
if input_xml != ""
  mpdict = xml2dict(input_xml, false)
  kwargs = Dict([Symbol(lowercase(k))=>lowercase(v) for (k,v) in mpdict])
  mp = MeasParams(; kwargs...)
else
  mp = MeasParams()
end

if "WALLTIMELIMIT" in keys(ENV)
  mp.walltimelimit = wtl2DateTime(ENV["WALLTIMELIMIT"], start_time)
end


try
  global rt = load(ARGS[1], "runstable")
catch
  error("Couldn't load runstable.")
end




# -------------------------------------------------------
#            Iterate runstable and measure
# -------------------------------------------------------
function foreachrun(mp::MeasParams, rt::DataFrame)

  for r in eachrow(rt) # for every selected run
    fpath = r[:PATH] # path to file
    cd(dirname(fpath))

    fpath = replace(fpath, ".in.xml", ".out.h5")
    if !isfile(fpath)
        fpath = replace(fpath, ".out.h5", ".out.h5.running")
        isfile(fpath) || continue
    end
    fpathnice = replace(fpath, "/projects/ag-trebst/bauer/", "")
    println(fpathnice); flush(STDOUT)

    outfile = replace(fpath, ".out.h5", ".meas.h5")

    if !endswith(outfile, ".running")
        # already measured? comment out to overwrite all measurement files.
        isfile(outfile) && begin println("Already measured. Skipping."); continue end

        # maybe job finished in the mean time. is there an old .meas.h5.running? if so, delete it.
        isfile(outfile*".running") && rm(outfile*".running")
    end

    try
        confs = ts_flat(fpath, "obs/configurations")
        R = 0
        WRITE_EVERY_NTH = 0
        delta_tau = 0.1
        h5open(fpath, "r") do f
          R = read(f["params/r"])
          delta_tau = read(f["params/delta_tau"])
          # L = read(f["params/L"])
          # B = read(f["params/beta"])
          
          # how many sweeps?
          WRITE_EVERY_NTH = read(f["params/write_every_nth"])
        end

        measure(mp, confs, R, delta_tau, WRITE_EVERY_NTH, outfile)
    catch err
        println("Failed. There was in issue for $(fpath). Maybe it doesn't have any configurations yet.")
        println(err)
        println()
    end

    println("Done.\n")
    flush(STDOUT)

    if now() >= mp.walltimelimit
      println("Approaching wall-time limit. Safely exiting.")
      exit(42)
    end
  end
end


function measure(mp::MeasParams, confs::AbstractArray{Float64, 4}, R::Float64, delta_tau::Float64, WRITE_EVERY_NTH::Int, outfile::String)
  num_confs = size(confs, ndims(confs))
  nsweeps = num_confs*WRITE_EVERY_NTH
  N = size(confs, 2)
  M = size(confs, 3)
  L = Int(sqrt(N))
  B = M * delta_tau

  # -------------------------------------------------------
  #            Allocate
  # -------------------------------------------------------
  # chi
  mp.chi_symm && (chi_dyn_symm = Observable(Array{Float64, 3}, "chi_dyn_symm"; alloc=num_confs))
  mp.chi && (chi_dyn = Observable(Array{Float64, 3}, "chi_dyn"; alloc=num_confs))
  # binder
  m2s = Vector{Float64}(num_confs)
  m4s = Vector{Float64}(num_confs)

  # -------------------------------------------------------
  #            Measure loop
  # -------------------------------------------------------
  mp.chi && println("Measuring chi_dyn/chi_dyn_symm/binder etc. ...");
  flush(STDOUT)

  @inbounds @views for i in 1:num_confs

      if mp.chi
        # chi
        chi = measure_chi_dynamic(confs[:,:,:,i])
        add!(chi_dyn, chi)

        if mp.chi_symm
          chi = (permutedims(chi, [2,1,3]) + chi)/2 # C4 is basically flipping qx and qy (which only go from 0 to pi since we perform a real fft.)
          add!(chi_dyn_symm, chi)
        end
      end

      if mp.binder
        # binder
        m = mean(confs[:,:,:,i],[2,3])
        m2s[i] = dot(m, m)
        m4s[i] = m2s[i]*m2s[i]
      end
  end

  # -------------------------------------------------------
  #            Postprocessing + Export
  # -------------------------------------------------------
  if mp.binder
    # binder postprocessing
    m2ev2 = mean(m2s)^2
    m4ev = mean(m4s)

    binder = Observable(Float64, "binder")
    add!(binder, m4ev/m2ev2)
  end

  # export results
  println("Calculating errors and exporting..."); flush(STDOUT)
  mp.chi && export_result(chi_dyn, outfile, "obs/chi_dyn"; timeseries=true)
  mp.chi_symm && export_result(chi_dyn_symm, outfile, "obs/chi_dyn_symm"; timeseries=true)

  mp.binder && export_result(binder, outfile, "obs/binder", error=false) # jackknife for error

  h5open(outfile, "r+") do fout
    HDF5.has(fout, "nsweeps") && HDF5.o_delete(fout, "nsweeps")
    HDF5.has(fout, "write_every_nth") && HDF5.o_delete(fout, "write_every_nth")
    fout["nsweeps"] = nsweeps
    fout["write_every_nth"] = WRITE_EVERY_NTH
  end
end







# -------------------------------------------------------
#            Main
# -------------------------------------------------------
# set num threads
try
  BLAS.set_num_threads(mp.num_threads)
  FFTW.set_num_threads(1)  = mp.num_threads
  ENV["OMP_NUM_THREADS"] = mp.num_threads
  ENV["MKL_NUM_THREADS"] = mp.num_threads
  ENV["JULIA_NUM_THREADS"] = mp.num_threads
end


pwd_before = pwd()
foreachrun(mp, rt)
cd(pwd_before)


end_time = now()
println("\nEnded: ", Dates.format(end_time, "d.u yyyy HH:MM"))
@printf("Duration: %.2f minutes", (end_time - start_time).value/1000./60.)
