using ProfileView
using Compat # BenchmarkTools bug for 0.6.1
using BenchmarkTools
using Helpers

start_time = now()
println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))

using Git
include("../parameters.jl")

### PROGRAM ARGUMENTS
ARGS = ["console_init.in.xml"]
if length(ARGS) == 1
  # ARGS = ["whatever.in.xml"]
  input_xml = ARGS[1]
  output_file = input_xml[1:searchindex(input_xml, ".in.xml")-1]*".out.h5"
elseif length(ARGS) == 2
  # Backward compatibility
  # ARGS = ["sdwO3_L_4_B_2_dt_0.1_2", 1]
  prefix = convert(String, ARGS[1])
  idx = 1
  try idx = parse(Int, ARGS[2]); end # SLURM_ARRAY_TASK_ID 
  output_file = prefix * ".task" * string(idx) * ".out.h5"

  println("Prefix is ", prefix, " and idx is ", idx)
  input_xml = prefix * ".task" * string(idx) * ".in.xml"
else
  error("Call with \"whatever.in.xml\" or e.g. \"sdwO3_L_4_B_2_dt_0.1_1 \${SLURM_ARRAY_TASK_ID}\"")
end

# hdf5 write test/ dump git commit
branch = Git.branch(dir=dirname(@__FILE__)).string[1:end-1]
if branch != "master"
  warn("Not on branch master but \"$(branch)\"!!!")
end
HDF5.h5open(output_file, "w") do f
  f["GIT_COMMIT_DQMC"] = Git.head(dir=dirname(@__FILE__)).string
  f["GIT_BRANCH_DQMC"] = branch
end


### PARAMETERS
p = Params()
p.output_file = output_file
xml2parameters!(p, input_xml)
parameters2hdf5(p, p.output_file)

println("HoppingType = ", HoppingType)
println("GreensType = ", GreensType)


include("../lattice.jl")
include("../stack.jl")
include("../linalg.jl")
include("../hoppings.jl")
include("../checkerboard.jl")
include("../interactions.jl")
include("../action.jl")
include("../local_updates.jl")
include("../global_updates.jl")
include("../observable.jl")
include("../boson_measurements.jl")
include("../fermion_measurements.jl")

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

srand(p.seed); # init RNG

# Init hsfield
println("\nInitializing HS field")
p.hsfield = rand(3,l.sites,p.slices)
println("Initializing boson action\n")
p.boson_action = calc_boson_action(p,l)


global const eye_flv = eye(p.flv,p.flv)
global const eye_full = eye(p.flv*l.sites,p.flv*l.sites)
global const ones_vec = ones(p.flv*l.sites)


# stack init and test
initialize_stack(s, p, l)
println("Building stack")
build_stack(s, p, l)
println("Initial propagate: ", s.current_slice, " ", s.direction)
propagate(s, p, l)