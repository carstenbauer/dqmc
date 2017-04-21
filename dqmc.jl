# dqmc.jl called with arguments: sdwO3_L_4_B_2_dt_0.1_1 ${SLURM_ARRAY_TASK_ID}
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
include("local_updates.jl")
include("global_updates.jl")
include("observable.jl")
include("boson_measurements.jl")
# include("tests/tests_gf_functions.jl")

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
f["params/TEST"] = 42
close(f)

# load parameters xml
params = Dict{Any, Any}()
try
  println("Prefix is ", prefix, " and idx is ", idx)
  params = xml2parameters(prefix * ".task" * string(idx) * ".in.xml")

  # Check and store code version (git commit)
  if haskey(params,"GIT_COMMIT_DQMC") && Git.head(dir=dirname(@__FILE__)) != params["GIT_COMMIT_DQMC"]
    warn("Git commit in input xml file does not match current commit of code.")
  end
  params["GIT_COMMIT_DQMC"] = Git.head(dir=dirname(@__FILE__))

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
p.c = parse(Float64, params["C"])
p.u = parse(Float64, params["U"])
p.global_updates = haskey(params,"GLOBAL_UPDATES")?parse(Bool, lowercase(params["GLOBAL_UPDATES"])):true;
p.beta = p.slices * p.delta_tau
p.flv = 4
if haskey(params,"BOX_HALF_LENGTH")
  p.box = Uniform(-parse(Float64, params["BOX_HALF_LENGTH"]),parse(Float64, params["BOX_HALF_LENGTH"]))
else
  p.box = Uniform(-0.2,0.2)
end
if haskey(params,"BOX_GLOBAL_HALF_LENGTH")
  p.box_global = Uniform(-parse(Float64, params["BOX_GLOBAL_HALF_LENGTH"]),parse(Float64, params["BOX_GLOBAL_HALF_LENGTH"]))
else
  p.box_global = Uniform(-0.1,0.1)
end
if haskey(params,"GLOBAL_RATE")
  p.global_rate = parse(Int64, params["GLOBAL_RATE"])
else
  p.global_rate = 5
end

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

# init hsfield
println("\nInitializing HS field")
p.hsfield = rand(3,l.sites,p.slices)
println("Initializing boson action\n")
p.boson_action = calculate_boson_action(p,l)

# stack init and test
s = Stack()
initialize_stack(s, p, l)
println("Building stack")
build_stack(s, p, l)
println("Initial propagate: ", s.current_slice, " ", s.direction)
propagate(s, p, l)

println("\nThermalization - ", p.thermalization)
acc_rate = 0.0
acc_rate_global = 0.0
prop_global = 0
acc_global = 0
tic()
for i in 1:p.thermalization

  for u in 1:2 * p.slices
    @inbounds propagate(s, p, l)

    if p.global_updates && (s.current_slice == p.slices && s.direction == -1 && mod(i, p.global_rate) == 0)
      # attempt global update after every fifth down-up sweep
      # println("Attempting global update...")
      prop_global += 1
      b = global_update(s, p, l)
      acc_rate_global += b
      acc_global += b
      # println("Accepted: ", b)
    end

    # println("Before local")
    # compare(s.greens, calculate_greens_udv_chkr(p,l,s.current_slice))
    acc_rate += local_updates(s, p, l)
    # println("Slice: ", s.current_slice, ", direction: ", s.direction, ", After local")
    # compare(s.greens, calculate_greens_udv_chkr(p,l,s.current_slice))
    # println("")
  end

  if mod(i, 10) == 0
    acc_rate = acc_rate / (10 * 2 * p.slices)
    acc_rate_global = acc_rate_global / (10 / p.global_rate)
    println("\t", i)
    @printf("\t\tup-down sweep dur: %.2fs\n", toq()/10)
    @printf("\t\tacc rate (local) : %.1f%%\n", acc_rate*100)
    if p.global_updates
      @printf("\t\tacc rate (global): %.1f%%\n", acc_rate_global*100)
      @printf("\t\tacc rate (global, overall): %.1f%%\n", acc_global/prop_global*100)
    end

    # adaption (first half of thermalization)
    if i < p.thermalization / 2 + 1
      if acc_rate < 0.5
        @printf("\t\tshrinking box: %.2f\n", 0.9*p.box.b)
        p.box = Uniform(-0.9*p.box.b,0.9*p.box.b)
      else
        @printf("\t\tenlarging box: %.2f\n", 1.1*p.box.b)
        p.box = Uniform(-1.1*p.box.b,1.1*p.box.b)
      end

      if p.global_updates
      if acc_global/prop_global < 0.5
        @printf("\t\tshrinking box_global: %.2f\n", 0.9*p.box_global.b)
        p.box_global = Uniform(-0.9*p.box_global.b,0.9*p.box_global.b)
      else
        @printf("\t\tenlarging box_global: %.2f\n", 1.1*p.box_global.b)
        p.box_global = Uniform(-1.1*p.box_global.b,1.1*p.box_global.b)
      end
      end
    end
    acc_rate = 0.0
    acc_rate_global = 0.0
    tic()
  end

end
toq();

println("")
initialize_stack(s, p, l)
build_stack(s, p, l)
println("Initial propagate: ", s.current_slice, " ", s.direction)
propagate(s, p, l)

println("\nMeasurements - ", p.measurements)
cs = min(p.measurements, 100)

configurations = Observable{Float64}("configurations", size(p.hsfield), cs)
# greens = Observable{Complex{Float64}}("greens", size(s.greens), cs)

boson_action = Observable{Float64}("boson_action", cs)
mean_abs_op = Observable{Float64}("mean_abs_op", cs)
mean_op = Observable{Float64}("mean_op", (3), cs)

acc_rate = 0.0
acc_rate_global = 0.0
tic()
for i in 1:p.measurements
  for u in 1:2 * p.slices
    @inbounds propagate(s, p, l)

    # GLOBAL UPDATE
    if p.global_updates && (s.current_slice == p.slices && s.direction == -1 && mod(i, p.global_rate) == 0)
      # attempt global update after every fifth down-up sweep
      # println("Attempting global update...")
      prop_global += 1
      b = global_update(s, p, l)
      acc_rate_global += b
      acc_global += b
      # println("Accepted: ", b)
    end

    acc_rate += local_updates(s, p, l)

    if s.current_slice == p.slices && s.direction == 1 # measure criterium
      # println("\t\tMeasuring")
      add_element(boson_action, p.boson_action)

      curr_mean_abs_op, curr_mean_op = measure_op(p.hsfield)
      add_element(mean_abs_op, curr_mean_abs_op)
      add_element(mean_op, curr_mean_op)

      add_element(configurations, p.hsfield)
      # add_element(greens, s.greens) # !! s.greens is only the effective Green's function !!

      if mod(i, cs) == 0
          println("Dumping...")
        @time begin
          confs2hdf5(output_file, configurations)
          obs2hdf5(output_file, greens)

          obs2hdf5(output_file, boson_action)
          obs2hdf5(output_file, mean_abs_op)
          obs2hdf5(output_file, mean_op)
          clear(boson_action)
          clear(mean_abs_op)
          clear(mean_op)

          clear(configurations)
          # clear(greens)
          println("Dumping block of $cs datapoints was a success")
        end
      end
    end
  end
  if mod(i, 10) == 0
    acc_rate = acc_rate / (10 * 2 * p.slices)
    acc_rate_global = acc_rate_global / (10 / p.global_rate)
    println("\t", i)
    @printf("\t\tup-down sweep dur: %.2fs\n", toq()/10)
    @printf("\t\tacc rate (local) : %.1f%%\n", acc_rate*100)
    if p.global_updates
      @printf("\t\tacc rate (global): %.1f%%\n", acc_rate_global*100)
      @printf("\t\tacc rate (global, overall): %.1f%%\n", acc_global/prop_global*100)
    end
    acc_rate = 0.0
    acc_rate_global = 0.0
    tic()
  end
end
toq();

# end # inbounds

end_time = now()
println("Ended: ", Dates.format(end_time, "d.u yyyy HH:MM"))
@printf("Duration: %.2f minutes", (end_time - start_time).value/1000./60.)
