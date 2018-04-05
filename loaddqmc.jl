# dqmc.jl called with arguments: whatever.in.xml
start_time = now()
println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))
println("Hostname: ", gethostname())

using Helpers
using Git
include("parameters.jl")

### PROGRAM ARGUMENTS
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
  error("Call with \"whatever.in.xml\" or e.g. \"sdwO3_L_4_B_2_dt_0.1_1 \${SLURM_ARRAY_TASK_ID}\"")
end

# hdf5 write test/ dump git commit
branch = Git.branch(dir=dirname(@__FILE__)).string[1:end-1]
if branch != "master"
  warn("Not on branch master but \"$(branch)\"!!!")
  flush(STDOUT)
end

### PARAMETERS
p = Parameters()
p.output_file = output_file
xml2parameters!(p, input_xml)

# mv old .out.h5 to .out.h5.running (and then try to resume below)
(isfile(output_file[1:end-8]) && !isfile(output_file)) && mv(output_file[1:end-8], output_file)

# check if there is a resumable running file
if isfile(output_file)
  try
    h5open(output_file) do f
      if HDF5.has(f, "resume") && read(f["count"]) > 0 && read(f["GIT_BRANCH_DQMC"]) == branch
        p.resume = true
      end
    end
  end
end
if !p.resume
  # overwrite (potential) running file
  h5open(output_file, "w") do f
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

  global const prevcount = h5read(output_file, "count")
  println("Found $(prevcount) configurations.")
  println()
  lastmeasurements = h5read(output_file, "params/measurements")
  global const prevmeasurements = prevcount * p.write_every_nth
  if prevmeasurements < lastmeasurements
    println("Finishing last run (i.e. overriding p.measurements)")
    p.measurements = lastmeasurements - prevmeasurements
  end
  global const lastconf = squeeze(h5read(output_file, "configurations", (:,:,:,prevcount)), 4)
  global const lastgreens = squeeze(h5read(output_file, "obs/greens/timeseries_real", (:,:,prevcount)) + 
                              im*h5read(output_file, "obs/greens/timeseries_imag", (:,:,prevcount)), 3)
end

println("HoppingType = ", HoppingType)
println("GreensType = ", GreensType)


include("lattice.jl")
include("stack.jl")
include("linalg.jl")
include("hoppings.jl")
if iseven(p.L)
  include("checkerboard.jl")
else
  include("checkerboard_generic.jl")
end
include("interactions.jl")
include("action.jl")
include("local_updates.jl")
include("global_updates.jl")
include("observable.jl")
include("boson_measurements.jl")
include("fermion_measurements.jl")

mutable struct Analysis
    acc_rate::Float64
    acc_rate_global::Float64
    prop_global::Int
    acc_global::Int

    Analysis() = new()
end

l = Lattice()
load_lattice(p,l)
s = Stack()
a = Analysis()

@printf("It took %.2f minutes to prepare everything. \n", (now() - start_time).value/1000./60.)

srand(p.seed); # init RNG

# Init hsfield
println("\nInitializing HS field")
p.hsfield = rand(p.opdim,l.sites,p.slices)
println("Initializing boson action\n")
p.boson_action = calculate_boson_action(p,l)

global const eye_flv = eye(p.flv,p.flv)
global const eye_full = eye(p.flv*l.sites,p.flv*l.sites)
global const ones_vec = ones(p.flv*l.sites)

# stack init and test
initialize_stack(s, p, l)
println("Building stack")
build_stack(s, p, l)
println("Initial propagate: ", s.current_slice, " ", s.direction)
propagate(s, p, l)
nothing