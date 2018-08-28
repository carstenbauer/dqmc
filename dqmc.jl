# dqmc.jl called with arguments: whatever.in.xml
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

try
  # Single core simulation
  BLAS.set_num_threads(1)
  ENV["OMP_NUM_THREADS"] = 1
  ENV["MKL_NUM_THREADS"] = 1
  ENV["JULIA_NUM_THREADS"] = 1
end


# -------------------------------------------------------
#             Input/Output file preparation
# -------------------------------------------------------
if length(ARGS) == 1
  # ARGS = ["whatever.in.xml"]
  input_xml = ARGS[1]
  output_file = input_xml[1:searchindex(input_xml, ".in.xml")-1]*".out.h5.running"
elseif length(ARGS) == 2
  # Backward compatibility
  # ARGS = ["sdwO3_L_4_B_2_dt_0.1_2", 1]
  prefix = convert(String, ARGS[1])
  idx = 1
  try idx = parse(Int, ARGS[2]); end # SLURM_ARRAY_TASK_ID 
  output_file = prefix * ".task" * string(idx) * ".out.h5.running"

  println("Prefix is ", prefix, " and idx is ", idx)
  input_xml = prefix * ".task" * string(idx) * ".in.xml"
else
  input_xml = "dqmc.in.xml"
  output_file = input_xml[1:searchindex(input_xml, ".in.xml")-1]*".out.h5.running"
end

# hdf5 write test/ dump git commit
using Git
branch = Git.branch(dir=dirname(@__FILE__)).string[1:end-1]
if branch != "master"
  println("!!!Not on branch master but \"$(branch)\"!!!")
  flush(STDOUT)
end

# TIMING parameter "hack"
using LightXML, Iterators
include("xml_parameters.jl")
params = xml2dict(input_xml, false)
haskey(params, "TIMING") && (parse(Bool, lowercase(params["TIMING"])) == true) && (global const TIMING = true)


# -------------------------------------------------------
#                    Includes
# -------------------------------------------------------
include("dqmc_framework.jl")
using JLD


# -------------------------------------------------------
#      Parse input parameters + check for resuming
# -------------------------------------------------------
p = Params()
p.output_file = output_file
xml2parameters!(p, input_xml)
if "WALLTIMELIMIT" in keys(ENV)
  p.walltimelimit = wtl2DateTime(ENV["WALLTIMELIMIT"], start_time)
end

# mv old .out.h5 to .out.h5.running (and then try to resume below)
(isfile(output_file[1:end-8]) && !isfile(output_file)) && mv(output_file[1:end-8], output_file)

# check if there is a resumable running file
if isfile(output_file)
  try
    jldopen(output_file) do f
      if HDF5.has(f.plain, "resume") && HDF5.has(f.plain, "obs/configurations/count") && read(f["obs/configurations/count"]) > 0 && read(f["GIT_BRANCH_DQMC"]) == branch
        p.resume = true
      end
    end
  end
end

if !p.resume

  # Start with a prethermalized conf?
  if isfile(p.output_file)
    jldopen(output_file) do f
      if HDF5.has(f.plain, "thermal_init")
        println("Using thermal_init/conf as starting configuration.")
        global const start_conf = read(f["thermal_init/conf"])

        if HDF5.has(f.plain, "thermal_init/prethermalized")
          p.prethermalized = read(f["thermal_init/prethermalized"])
          println("Using thermal_init/prethermalized. Will do $(p.thermalization - p.prethermalized) ud-sweeps.")
        end

        if HDF5.has(f.plain, "thermal_init/box")
          box = read(f, "thermal_init/box")
          p.box = Uniform(-box, box)
          println("Using thermal_init/box.")
        end

        if HDF5.has(f.plain, "thermal_init/box_global")
          box_global = read(f, "thermal_init/box_global")
          p.box_global = Uniform(-box_global, box_global)
          println("Using thermal_init/box_global.")
        end
      end
    end
  end

  # overwrite (potential) running file
  jldopen(output_file, "w") do f
    f["GIT_COMMIT_DQMC"] = Git.head(dir=dirname(@__FILE__)).string
    f["GIT_BRANCH_DQMC"] = branch
    f["RUNNING"] = 1
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
  end
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
@printf("It took %.2f minutes to prepare everything. \n", (now() - start_time).value/1000./60.)

if !mc.p.resume
  if isdefined(:start_conf)
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
mv(output_file, output_file[1:end-8], remove_destination=true) # .out.h5.running -> .out.h5

end_time = now()
println("Ended: ", Dates.format(end_time, "d.u yyyy HH:MM"))
@printf("Duration: %.2f minutes", (end_time - start_time).value/1000./60.)
