using ProfileView
using Compat # BenchmarkTools bug for 0.6.1
using BenchmarkTools
using Helpers

start_time = now()
println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))

using Helpers
include("../parameters.jl")

### PROGRAM ARGUMENTS
# ARGS = ["sdwO3_L_4_B_2_dt_0.1_2", 1]
prefix = convert(String, ARGS[1])
idx = 1
try idx = parse(Int, ARGS[2]); end # SLURM_ARRAY_TASK_ID 
output_file = prefix * ".task" * string(idx) * ".out.h5"

println("Prefix is ", prefix, " and idx is ", idx)
input_xml = prefix * ".task" * string(idx) * ".in.xml"

# hdf5 write test
f = HDF5.h5open(output_file, "w")
f["params/TEST"] = 42
close(f)


### PARAMETERS
p = Parameters()
p.output_file = output_file
params = parse_inputxml(p, input_xml)

### SET DATATYPES
global const HoppingType = p.Bfield ? Complex128 : Float64;
global const GreensType = Complex128;
println("HoppingType = ", HoppingType)
println("GreensType = ", GreensType)


include("../linalg.jl")
include("../lattice.jl")
include("../checkerboard.jl")
include("../interactions.jl")
include("../action.jl")
include("../stack.jl")
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


### LATTICE
l = Lattice()
l.L = parse(Int, p.lattice_file[end-4])
l.t = reshape([parse(Float64, f) for f in split(params["HOPPINGS"], ',')],(2,2))
init_lattice_from_filename(params["LATTICE_FILE"], l)
init_neighbors_table(p,l)
init_time_neighbors_table(p,l)
if p.Bfield
  init_hopping_matrix_exp_Bfield(p,l)
  init_checkerboard_matrices_Bfield(p,l)
else
  init_hopping_matrix_exp(p,l)
  init_checkerboard_matrices(p,l)
end

s = Stack()
a = Analysis()

preallocate_arrays(p,l.sites)

# Init hsfield
println("\nInitializing HS field")
p.hsfield = rand(3,l.sites,p.slices)
println("Initializing boson action\n")
p.boson_action = calculate_boson_action(p,l)


# stack init and test
initialize_stack(s, p, l)
println("Building stack")
build_stack(s, p, l)
println("Initial propagate: ", s.current_slice, " ", s.direction)
propagate(s, p, l)