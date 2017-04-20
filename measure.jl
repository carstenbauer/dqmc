# measure.jl called with arguments: inputfile.out.h5
# expects additional file inputfile.measure.in.xml
start_time = now()
println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))

using Helpers
using Git
# include("linalg.jl")
include("parameters.jl")
include("xml_parameters.jl")
include("lattice.jl")
include("checkerboard.jl")
# include("interactions.jl")
include("action.jl")
# include("stack.jl")
# include("local_updates.jl")
# include("global_updates.jl")
include("observable.jl")
include("boson_measurements.jl")
include("fermion_measurements.jl")
# include("tests/tests_gf_functions.jl")

# ARGS = ["sdwO3_L_4_B_2_dt_0.1_1.task1.out.h5"]
output_file = convert(String, ARGS[1])

# load measure parameters xml
params = Dict{Any, Any}()
try
  params = xml2parameters(output_file[1:end-7] * ".measure.in.xml")
catch e
  println(e)
end

# hdf5 read/write test
f = HDF5.h5open(output_file, "r+")
if HDF5.has(f, "params/TEST") HDF5.o_delete(f["params"],"TEST") end
f["params/TEST"] = 43

# Check and store code version (git commit)
if Git.head(dir=dirname(@__FILE__)) != f["params/GIT_COMMIT"]
  warn("Git commit in h5 file does not match current commit of code.")
end

p = Parameters()
p.thermalization = parse(Int64, read(f["params/THERMALIZATION"]))
p.measurements = parse(Int64, read(f["params/MEASUREMENTS"]))
p.slices = parse(Int64, read(f["params/SLICES"]))
p.delta_tau = parse(Float64, read(f["params/DELTA_TAU"]))
p.safe_mult = parse(Int64, read(f["params/SAFE_MULT"]))
p.lattice_file = read(f["params/LATTICE_FILE"])
p.mu = parse(Float64, read(f["params/MU"]))
p.lambda = parse(Float64, read(f["params/LAMBDA"]))
p.r = parse(Float64, read(f["params/R"]))
p.c = parse(Float64, read(f["params/C"]))
p.u = parse(Float64, read(f["params/U"]))
p.global_updates = parse(Bool, lowercase(read(f["params/GLOBAL_UPDATES"])))
p.beta = p.slices * p.delta_tau
p.flv = 4
p.box = Uniform(-parse(Float64, read(f["params/BOX_HALF_LENGTH"])),parse(Float64, read(f["params/BOX_HALF_LENGTH"])))
p.box_global = Uniform(-parse(Float64, read(f["params/BOX_GLOBAL_HALF_LENGTH"])),parse(Float64, read(f["params/BOX_GLOBAL_HALF_LENGTH"])))
p.global_rate = parse(Int64, read(f["params/GLOBAL_RATE"]))

# load lattice xml and prepare hopping matrices
l = Lattice()
# OPT: better filename parsing
Lpos = maximum(search(p.lattice_file,"L_"))+1
l.L = parse(Int, p.lattice_file[Lpos:Lpos+minimum(search(p.lattice_file[Lpos:end],"_"))-2])
l.t = reshape([parse(Float64, f) for f in split(params["HOPPINGS"], ',')],(2,2))
init_lattice_from_filename(params["LATTICE_FILE"], l)
println("Initializing neighbor-tables")
init_neighbors_table(p,l)
init_time_neighbors_table(p,l)
println("Initializing hopping exponentials")
init_hopping_matrix_exp(p,l)
init_checkerboard_matrices(p,l)

# # init hsfield
# println("\nInitializing HS field")
# p.hsfield = rand(3,l.sites,p.slices)
# println("Initializing boson action\n")
# p.boson_action = calculate_boson_action(p,l)

# # stack init and test
# s = Stack()
# initialize_stack(s, p, l)
# println("Building stack")
# build_stack(s, p, l)
# println("Initial propagate: ", s.current_slice, " ", s.direction)
# propagate(s, p, l)

close(f)



end_time = now()
println("Ended: ", Dates.format(end_time, "d.u yyyy HH:MM"))
@printf("Duration: %.2f minutes", (end_time - start_time).value/1000./60.)
