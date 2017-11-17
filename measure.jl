# measure.jl called with argument: inputfile.out.h5 OR inputfile.measure.in.xml
# expects additional file inputfile.measure.in.xml OR inputfile.out.h5
start_time = now()
println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))

using Helpers
using Git
include("parameters.jl")
include("xml_parameters.jl")

### PROGRAM ARGUMENTS
output_file = convert(String, ARGS[1])
input_file = output_file[1:end-7] * ".measure.in.xml"
if output_file[end-2:end] == "xml"
  input_file = output_file
  output_file = input_file[1:end-15] * ".out.h5"
end

# test input/output files
if !isfile(input_file) || !isfile(output_file)
  error("Couldn't find both .measure.in.xml and .out.h5 input/output files.")
end

# hdf5 read/write test
f = HDF5.h5open(output_file, "r+")
if HDF5.has(f, "params/TEST") HDF5.o_delete(f["params"],"TEST") end
f["params/TEST"] = 43

existing_obs = listobs(output_file)

type Measurements
  boson_suscept::Bool
  binder::Bool
  greens::Bool

  safe_mult::Int

  overwrite::Bool

  function Measurements()
    m = new()
    m.boson_suscept = true
    m.binder = false
    m.greens = false

    m.safe_mult = -1
    m.overwrite = true
    return m
  end
end

m = Measurements()

# load parameters from measure.in.xml
params = Dict{Any, Any}()
try
  params = xml2parameters(input_file)

  if haskey(params, "OVERWRITE") && !parse(Bool, lowercase(params["OVERWRITE"]))
    m.overwrite = false
  end

  if haskey(params, "BOSON_SUSCEPT") && parse(Bool, lowercase(params["BOSON_SUSCEPT"]))
    m.boson_suscept = m.overwrite || !("boson_suscept" in existing_obs)
    if m.boson_suscept
      delete(output_file, "boson_suscept")
    end
  end
  if haskey(params, "BINDER") && parse(Bool, lowercase(params["BINDER"]))
    m.binder = m.overwrite || !("binder" in existing_obs)
    if m.binder
      delete(output_file, "m2")
      delete(output_file, "m4")
    end
  end
  if haskey(params, "GREENS") && parse(Bool, lowercase(params["GREENS"]))
    m.greens = m.overwrite || !("greens" in existing_obs)

    if m.greens
      delete(output_file, "greens")
      delete(output_file, "fermion_action")
      delete(output_file, "action")
    end
  end
  if haskey(params, "SAFE_MULT")
    m.safe_mult = parse(Int64, params["SAFE_MULT"])
  end
catch e
  println(e)
  exit()
end


### CHECK GIT COMMIT CONSISTENCY
git_commit = Git.head(dir=dirname(@__FILE__))
if !haskey(f, "params/GIT_COMMIT_DQMC") || (git_commit != f["params/GIT_COMMIT_DQMC"])
  warn("Git commit used for dqmc doesn't match current commit of code!!!")
end
if HDF5.exists(f, "params/GIT_COMMIT_MEASURE")
  warn("Overwriting GIT_COMMIT_MEASURE!")
  HDF5.o_delete(f, "params/GIT_COMMIT_MEASURE")
end
f["params/GIT_COMMIT_MEASURE"] = git_commit


# load configurations
confs = read(f["configurations"])
num_confs = size(confs)[end]

close(f)


######### All set. Let's get going ##########

include("lattice.jl")
include("stack.jl")
include("linalg.jl")
include("checkerboard.jl")
include("interactions.jl")
include("action.jl")
# include("local_updates.jl")
# include("global_updates.jl")
include("observable.jl")
include("boson_measurements.jl")
include("fermion_measurements.jl")

# load DQMC simulation parameters
p = Parameters()
load_parameters_h5(p, output_file)

### LATTICE
l = Lattice()
l.L = parse(Int, p.lattice_file[findlast(collect(p.lattice_file), '_')+1:end-4])
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
if m.binder
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
