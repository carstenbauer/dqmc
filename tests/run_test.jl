using Helpers
include("../linalg.jl")
include("../parameters.jl")
include("../xml_parameters.jl")
include("../lattice.jl")
include("../checkerboard.jl")
include("../interactions.jl")
include("../action.jl")
include("../stack.jl")
include("../updates.jl")
include("../observable.jl")
include("../measurements.jl")

# @inbounds begin
# ARGS = ["sdwO3_L_4_B_2_dt_0.1_1", 1]
prefix = convert(String, ARGS[1])
idx = 1
try
  idx = parse(Int, ARGS[2]) # SLURM_ARRAY_TASK_ID
end
output_file = prefix * ".task" * string(idx) * ".out.h5"

# hdf5 write test
f = HDF5.h5open(output_file, "w")
f["parameters/TEST"] = 42
close(f)

# load parameters xml
params = Dict{Any, Any}()
try
  params = xml2parameters(prefix, idx)
  parameters2hdf5(params, output_file)
catch e
  println(e)
end


p = Parameters()
p.thermalization = parse(Int, params["THERMALIZATION"])
p.measurements = parse(Int, params["MEASUREMENTS"])
p.slices = parse(Int, params["SLICES"])
p.delta_tau = parse(Float64, params["DELTA_TAU"])
p.safe_mult = parse(Int, params["SAFE_MULT"])
p.lattice_file = params["LATTICE_FILE"]
srand(parse(Int, params["SEED"]))
p.mu = parse(Float64, params["MU"])
p.lambda = parse(Float64, params["LAMBDA"])
p.r = parse(Float64, params["R"])
p.u = parse(Float64, params["U"])
p.beta = p.slices * p.delta_tau
p.flv = 4
if haskey(params,"BOX_HALF_LENGTH")
  p.box = Uniform(-parse(Float64, params["BOX_HALF_LENGTH"]),parse(Float64, params["BOX_HALF_LENGTH"]))
else
  p.box = Uniform(-0.2,0.2)
end

# load lattice xml and prepare hopping matrices
l = Lattice()
# OPT: better filename parsing
Lpos = maximum(search(p.lattice_file,"L_"))+1
l.L = parse(Int, p.lattice_file[Lpos:Lpos+minimum(search(p.lattice_file[Lpos:end],"_"))-2])
l.t = reshape([parse(Float64, f) for f in split(params["HOPPINGS"], ',')],(2,2))
init_lattice_from_filename(params["LATTICE_FILE"], l)
println("Initializing neighbor-tables")
@time init_neighbors_table(p,l)
@time init_time_neighbors_table(p,l)
println("Initializing hopping exponentials")
@time init_hopping_matrix_exp(p,l)
@time init_checkerboard_matrices(p,l)

# init hsfield
println("\nInitializing HS field")
@time p.hsfield = rand(3,l.sites,p.slices)
println("\nPrecalculating sinh cosh terms")
@time init_interaction_sinh_cosh(p,l)
println("Initializing boson action\n")
@time calculate_boson_action(p,l)

# stack init
s = Stack()
initialize_stack(s, p, l)

@time build_stack(s, p, l)
println("Propagate ", s.current_slice, " ", s.direction)
propagate(s, p, l)

# testing
include("tests_gf.jl")
plot_gf_error_propagation(s,p,l,"exact")

# include("tests_gf_stabilization.jl")
# plot_svs_of_slice_matrix_chain(p,l)
