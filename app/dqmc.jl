# dqmc.jl called with arguments: whatever.in.xml
using Dates, LinearAlgebra
start_time = now()
println("\nStarted: ", Dates.format(start_time, "d.u yyyy HH:MM"))
println("Hostname: ", gethostname())


try
  nthreads = parse(Int, ENV["OMP_NUM_THREADS"])
  LinearAlgebra.BLAS.set_num_threads(nthreads)
  ENV["MKL_NUM_THREADS"] = nthreads
  ENV["JULIA_NUM_THREADS"] = 1
  println("Using $(nthreads) threads for BLAS, OMP, and MKL.")
catch err
  # Single core simulation
  LinearAlgebra.BLAS.set_num_threads(1)
  ENV["OMP_NUM_THREADS"] = 1
  ENV["MKL_NUM_THREADS"] = 1
  ENV["JULIA_NUM_THREADS"] = 1
  println("Using single thread for BLAS, OMP, and MKL.")
end


# -------------------------------------------------------
#             Input/Output file preparation
# -------------------------------------------------------
if length(ARGS) == 1
  # ARGS = ["whatever.in.xml"]
  input_xml = ARGS[1]
  output_file = input_xml[1:first(findfirst(".in.xml", input_xml))-1]*".out.h5.running"
elseif length(ARGS) == 2
  # Backward compatibility
  # ARGS = ["sdwO3_L_4_B_2_dt_0.1_2", 1]
  prefix = convert(String, ARGS[1])
  idx = 1
  try global idx = parse(Int, ARGS[2]); catch end # SLURM_ARRAY_TASK_ID 
  output_file = prefix * ".task" * string(idx) * ".out.h5.running"

  println("Prefix is ", prefix, " and idx is ", idx)
  input_xml = prefix * ".task" * string(idx) * ".in.xml"
else
  input_xml = "dqmc.in.xml"
  output_file = input_xml[1:first(findfirst(".in.xml", input_xml))-1]*".out.h5.running"
end
@assert isfile(input_xml)

# hdf5 write test/ dump git commit
using Git
branch = Git.branch(dir=dirname(@__FILE__)).string[1:end-1]
if branch != "master"
  println("!!!Not on branch master but \"$(branch)\"!!!")
  flush(stdout)
end

# TIMING parameter "hack"
using LightXML #, Iterators
include("../src/xml_parameters.jl")
params = xml2dict(input_xml, false)
haskey(params, "TIMING") && (parse(Bool, lowercase(params["TIMING"])) == true) && (global const TIMING = true)


# -------------------------------------------------------
#                    Includes
# -------------------------------------------------------
include("../src/dqmc_framework.jl")
using JLD


# -------------------------------------------------------
#      Parse input parameters + check for resuming
# -------------------------------------------------------
p = Params()
p.output_file = output_file
xml2parameters!(p, input_xml)
set_walltimelimit!(p, start_time)

# mv old .out.h5 to .out.h5.running (and then try to resume below)
# (isfile(output_file[1:end-8]) && !isfile(output_file)) && mv(output_file[1:end-8], output_file)
isfile(output_file[1:end-8]) && begin println("Nothing to do here. Found .out.h5 file."); exit(); end

# check if there is a resumable running file
if isfile(output_file)
  println("\nFound old output file. Let's see if we can resume.")
  alreadydone = false
  resumable = false
  try
    jldopen(output_file) do f
      if HDF5.has(f.plain, "resume") && HDF5.has(f.plain, "obs/configurations/count") && read(f["GIT_BRANCH_DQMC"]) == branch
        nconfs = read(f["obs/configurations/count"])
        pmeasurements = read(f["params/measurements"])
        pwrite_every_nth = read(f["params/write_every_nth"])
        measurements = nconfs * pwrite_every_nth

        (measurements >= pmeasurements) && (measurements >= p.measurements) && (global alreadydone = true)
        (nconfs > 0) && (global resumable = true)
      end
    end
  catch
  end

  if alreadydone
    mv(output_file, output_file[1:end-8])
    println("Nothing to do here. There are already enough configurations present.")
    exit();
  else
    if resumable
      println("Will resume.")
      p.resume = true
    else
      println("Couldn't resume.")
    end
  end
end

if !p.resume

  # Start with a prethermalized conf?
  if isfile(p.output_file)
    try
      jldopen(output_file) do f
        if HDF5.has(f.plain, "thermal_init")
          println("Using thermal_init/conf as starting configuration.")
          global start_conf = read(f["thermal_init/conf"])

          if HDF5.has(f.plain, "thermal_init/prethermalized")
            p.prethermalized = read(f["thermal_init/prethermalized"])
            println("Using thermal_init/prethermalized. Will do $(p.thermalization - p.prethermalized) ud-sweeps.")
          end

          if HDF5.has(f.plain, "thermal_init/box")
            box = read(f, "thermal_init/box")
            p.box = box
            println("Using thermal_init/box.")
          end

          if HDF5.has(f.plain, "thermal_init/box_global")
            box_global = read(f, "thermal_init/box_global")
            p.box_global = box_global
            println("Using thermal_init/box_global.")
          end
        end
      end
    catch
    end
  end

  # overwrite (potential) running file
  jldopen(output_file, isfile(p.output_file) ? "r+" : "w") do f
    HDF5.has(f.plain, "GIT_COMMIT_DQMC") && o_delete(f.plain, "GIT_COMMIT_DQMC")
    HDF5.has(f.plain, "GIT_BRANCH_DQMC") && o_delete(f.plain, "GIT_BRANCH_DQMC")
    HDF5.has(f.plain, "RUNNING") && o_delete(f.plain, "RUNNING")
    f["GIT_COMMIT_DQMC"] = Git.head(dir=dirname(@__FILE__)).string
    f["GIT_BRANCH_DQMC"] = branch
    f["RUNNING"] = 1

    # we will dump the parameters below
    HDF5.has(f.plain, "params") && o_delete(f.plain, "params")
    HDF5.has(f.plain, "obs") && o_delete(f.plain, "obs")
  end
end

if !p.resume
  parameters2hdf5(p, p.output_file)
else
  println()
  println("RESUMING MODE -----------------")
  h5open(output_file, "r+") do f
    !HDF5.has(f, "RUNNING") || HDF5.o_delete(f, "RUNNING")
    f["RUNNING"] = 1
    !HDF5.has(f, "RESUME") || HDF5.o_delete(f, "RESUME")
    f["RESUME"] = 1
  end

  confs = loadobs_frommemory(output_file, "obs/configurations")
  prevcount = length(confs)
  println("Found $(prevcount) configurations.")
  println()
  lastmeasurements = h5read(output_file, "params/measurements")
  global prevmeasurements = prevcount * p.write_every_nth
  if prevmeasurements < lastmeasurements
    println("Finishing last run (i.e. overriding p.measurements)")
    p.measurements = lastmeasurements - prevmeasurements
  elseif prevmeasurements < p.measurements
    println("Extension run.")
    p.measurements = p.measurements - prevmeasurements
  end
  @show prevmeasurements
  @show p.measurements
  global lastconf = confs[end]
end



# -------------------------------------------------------
#                      Simulation
# -------------------------------------------------------
mc = DQMC(p)

println()
println("HoppingEltype = ", heltype(mc))
println("GreensEltype = ", geltype(mc))
println()
@printf("It took %.2f minutes to prepare everything. \n", (now() - start_time).value/1000/60)

if !mc.p.resume
  if @isdefined start_conf
    init!(mc, start_conf)
  else
    init!(mc)
  end
  run!(mc)
else
  resume!(mc, lastconf, prevmeasurements)
end

h5open(output_file, "r+") do f
  HDF5.o_delete(f, "RUNNING")
  f["RUNNING"] = 0
end
mv(output_file, output_file[1:end-8], force=true) # .out.h5.running -> .out.h5

end_time = now()
println("Ended: ", Dates.format(end_time, "d.u yyyy HH:MM"))
@printf("Duration: %.2f minutes", (end_time - start_time).value/1000/60)
