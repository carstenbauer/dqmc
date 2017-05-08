# measure.jl called with argument: inputfile.out.h5 OR inputfile.measure.in.xml
# expects additional file inputfile.measure.in.xml OR inputfile.out.h5
start_time = now()
println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))

using Helpers
using Git
include("linalg.jl")
include("parameters.jl")
include("xml_parameters.jl")
include("lattice.jl")
include("checkerboard.jl")
include("interactions.jl")
include("action.jl")
include("stack.jl")
# include("local_updates.jl")
# include("global_updates.jl")
include("observable.jl")
include("boson_measurements.jl")
include("fermion_measurements.jl")

# ARGS = ["sdwO3_L_4_B_2_dt_0.1_1.task1.out.h5"]
output_file = convert(String, ARGS[1])
input_file = output_file[1:end-7] * ".measure.in.xml"
if output_file[end-2:end] == "xml"
  input_file = output_file
  output_file = input_file[1:end-15] * ".out.h5"
end

# load parameters from measure.in.xml
BOSON_SUSCEPT = false
BINDER_CUMULANT = false
GREENS_FUNCTION = false
GREENS_FUNCTION_SAFE_MULT = 1
params = Dict{Any, Any}()
try
  params = xml2parameters(input_file)

  if haskey(params, "BOSON_SUSCEPT") && parse(Bool, lowercase(params["BOSON_SUSCEPT"]))
    BOSON_SUSCEPT = true
    delete(output_file, "boson_suscept")
  end
  if haskey(params, "BINDER_CUMULANT") && parse(Bool, lowercase(params["BINDER_CUMULANT"]))
    BINDER_CUMULANT = true
    delete(output_file, "m2")
    delete(output_file, "m4")
  end
  if haskey(params, "GREENS_FUNCTION") && parse(Bool, lowercase(params["GREENS_FUNCTION"]))
    GREENS_FUNCTION = true
    delete(output_file, "greens")
    delete(output_file, "fermion_action")
    delete(output_file, "action")
  end
  if haskey(params, "GREENS_FUNCTION_SAFE_MULT")
    GREENS_FUNCTION_SAFE_MULT = parse(Int64, params["GREENS_FUNCTION_SAFE_MULT"])
  end
catch e
  println(e)
end

# hdf5 read/write test
f = HDF5.h5open(output_file, "r+")
if HDF5.has(f, "params/TEST") HDF5.o_delete(f["params"],"TEST") end
f["params/TEST"] = 43

# Check and store code version (git commit)
git_commit = Git.head(dir=dirname(@__FILE__))
if git_commit != f["params/GIT_COMMIT_DQMC"]
  info("Git commit used for dqmc does not match current commit of code.")
end
if HDF5.exists(f, "params/GIT_COMMIT_MEASURE")
  info("Has been measured before!?")
  HDF5.o_delete(f, "params/GIT_COMMIT_MEASURE")
end
f["params/GIT_COMMIT_MEASURE"] = git_commit

# load DQMC simulation parameters
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
l.t = reshape([parse(Float64, f) for f in split(read(f["params/HOPPINGS"]), ',')],(2,2))
init_lattice_from_filename(p.lattice_file, l)
println("Initializing neighbor-tables")
init_neighbors_table(p,l)
init_time_neighbors_table(p,l)
println("Initializing hopping exponentials")
init_hopping_matrix_exp(p,l)
init_checkerboard_matrices(p,l)


# load configurations
confs = read(f["configurations"])
num_confs = size(confs)[end]
if read(f["count"]) != num_confs
  warn("number of configurations found in .out.h5 file does not match stated count value")
end

close(f)

# measure
tic()
println("\nMeasuring...")
cs = num_confs # currently, we do not dump intermediate results

chi = Observable{Float64}("boson_suscept", cs)
m2 = Observable{Float64}("m2", cs)
m4 = Observable{Float64}("m4", cs)

greens = Observable{Complex128}("greens", (p.flv*l.sites, p.flv*l.sites), cs)
fermion_action = Observable{Float64}("fermion_action", cs)
action = Observable{Float64}("action", cs)

S_b = hdf52obs(output_file, "boson_action").timeseries

elapsed_time = 0.0
for c in 1:num_confs
  conf = confs[:,:,:,c]

  if BINDER_CUMULANT
    curr_m2, curr_m4 = measure_binder_factors(conf)
    add_element(m2, curr_m2)
    add_element(m4, curr_m4)
  end

  if BOSON_SUSCEPT
    add_element(chi, measure_chi_static_direct(conf))
  end

  if GREENS_FUNCTION
    p.hsfield = conf
    curr_greens, S_f = measure_greens_and_logdet(p, l, GREENS_FUNCTION_SAFE_MULT)
    add_element(greens, curr_greens)
    add_element(fermion_action, S_f) # log(det(G)) = S_f
    add_element(action, S_b[c]+S_f)
  end

  if mod(c, 100) == 0
    println("\t", c)
    tperconf = toq()/100
    elapsed_time += tperconf*100
    @printf("\t\ttime per conf: %.2fs\n", tperconf)
    @printf("\t\ttime elapsed: %.2fs\n", elapsed_time)
    @printf("\t\testimated time remaining: %.2fs\n", (num_confs - c) * tperconf)
    tic()
  end
end

# save to disk
println("Dumping results...")
@time begin
  obs2hdf5(output_file, chi)
  obs2hdf5(output_file, m2)
  obs2hdf5(output_file, m4)

  obs2hdf5(output_file, greens)
  obs2hdf5(output_file, fermion_action)
  obs2hdf5(output_file, action)

  println("Dumping block of $cs datapoints was a success")
end

# calculate Binder cumulant
if BINDER_CUMULANT
  println("\nCalculating and saving Binder cumulant")
  m2ev2 = mean(m2.timeseries)^2
  m4ev = mean(m4.timeseries)
  binder = 1 - 5/3 * m4ev/m2ev2
  f = h5open(output_file, "r+")
    if HDF5.exists(f, "obs/binder") HDF5.o_delete(f, "obs/binder") end
    f["obs/binder"] = binder
  close(f)
end




end_time = now()
println("\nEnded: ", Dates.format(end_time, "d.u yyyy HH:MM"))
@printf("Duration: %.2f minutes", (end_time - start_time).value/1000./60.)
