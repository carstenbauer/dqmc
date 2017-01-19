# dqmc.jl called with arguments: sdwO3_L_4_B_2_dt_0.1_1 ${SLURM_ARRAY_TASK_ID}
include("parameters.jl")
include("xml_parameters.jl")
include("lattice.jl")
include("helpers.jl")

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
l = lattice()
l.L = parse(Int, p.lattice_file[maximum(search(p.lattice_file,"L_"))+1])

l.t = hoppings([parse(Float64, f) for f in split(params["HOPPINGS"], ',')]...)
println("Hoppings are ", l.t)

init_lattice_from_filename(params["LATTICE_FILE"], l)
l.urneighbors = zeros(Int64, 2, l.sites) # up and right neighbor
for i in 1:l.sites
  bonds = l.bonds[l.site_bonds[i,:],:]
  l.urneighbors[:,i] = bonds[find(e->e==i,bonds[:,1]),2]
end

# init_checkerboard_matrices(l, p.delta_tau)
# free_fermion_wavefunction(p, l)

p.hsfield = rand(3,l.sites,p.slices)
