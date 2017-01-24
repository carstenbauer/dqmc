# dqmc.jl called with arguments: sdwO3_L_4_B_2_dt_0.1_1 ${SLURM_ARRAY_TASK_ID}
include("helpers.jl")
include("linalg.jl")
include("parameters.jl")
include("xml_parameters.jl")
include("lattice.jl")
include("interactions.jl")
include("stack.jl")

 # ARGS = ["sdwO3_L_4_B_2_dt_0.1_1", 1]
prefix = convert(String, ARGS[1])
idx = 1
try
  idx = parse(Int, ARGS[2]) # SLURM_ARRAY_TASK_ID
end
output_file = prefix * ".task" * string(idx) * ".out.h5"

if !isfile(output_file)
  f = HDF5.h5open(output_file, "w")
  f["parameters/TEST"] = 42
  close(f)
end

# load parameters xml
params = Dict{Any, Any}()
try
  params = xml2parameters(prefix, idx)
  parameters2hdf5(params, output_file)
catch e
  println(e)
end

p = parameters()
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

# load lattice xml and prepare hopping matrices
l = lattice()
# OPT: better filename parsing
Lpos = maximum(search(p.lattice_file,"L_"))+1
l.L = parse(Int, p.lattice_file[Lpos:Lpos+minimum(search(p.lattice_file[Lpos:end],"_"))-2])
l.t = hoppings([parse(Float64, f) for f in split(params["HOPPINGS"], ',')]...)
println("Hoppings are ", str(l.t))
init_lattice_from_filename(params["LATTICE_FILE"], l)
l.urneighbors = zeros(Int64, 2, l.sites)
for i in 1:l.sites
  bofi = l.bonds[l.site_bonds[i,:],:]
  l.urneighbors[:,i] = bofi[findin(bofi[:,1],i),2]
end
init_hopping_matrix(p,l)
# init_checkerboard_matrices(l, p.delta_tau)

# init hsfield
println("Initializing HS field...")
@time p.hsfield = rand(3,l.sites,p.slices)

# stack init and test
s = stack()
initialize_stack(s, p, l)


@time build_stack(s, p, l)
# # @time begin for __ in 1:(2 * p.slices) propagate(s, p, l); flipped = simple_update(s, p, l); end end

# println("Propagate ", s.current_slice, " ", s.direction)
# propagate(s, p, l)
# println(real(diag(s.greens)))
# # simple_update(s, p, l)
# updated_greens = copy(s.greens)
# build_stack(s, p, l)
# propagate(s, p, l)
# println("Update test\t", maximum(abs(updated_greens - s.greens)))
