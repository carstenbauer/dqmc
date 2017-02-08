# dqmc.jl called with arguments: sdwO3_L_4_B_2_dt_0.1_1 ${SLURM_ARRAY_TASK_ID}
using Helpers
include("linalg.jl")
include("parameters.jl")
include("xml_parameters.jl")
include("lattice.jl")
include("interactions.jl")
include("action.jl")
include("stack.jl")
include("updates.jl")
include("observable.jl")
include("measurements.jl")

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
l.t = Hoppings([parse(Float64, f) for f in split(params["HOPPINGS"], ',')]...)
println("Hoppings are ", str(l.t))
init_lattice_from_filename(params["LATTICE_FILE"], l)
println("Initializing neighbor-tables")
@time init_neighbors_table(p,l)
@time init_time_neighbors_table(p,l)
init_hopping_matrix(p,l)
# init_checkerboard_matrices(l, p.delta_tau)

# init hsfield
println("\nInitializing HS field")
@time p.hsfield = rand(3,l.sites,p.slices)
println("Initializing boson action\n")
calculate_boson_action(p,l)

# stack init and test
s = Stack()
initialize_stack(s, p, l)
@time build_stack(s, p, l)
# # @time begin for __ in 1:(2 * p.slices) propagate(s, p, l); flipped = simple_update(s, p, l); end end

# println("Propagate ", s.current_slice, " ", s.direction)
# propagate(s, p, l)
# println(real(diag(s.greens)))
# local_updates(s, p, l)
# updated_greens = copy(s.greens)
@time build_stack(s, p, l)
println("Propagate ", s.current_slice, " ", s.direction)
propagate(s, p, l)
# println("Update test\t", maximum(abs(updated_greens - s.greens)))


println("\nThermalization - ", p.thermalization)
acc_rate = 0.0
tic()
for i in 1:p.thermalization

  for u in 1:2 * p.slices
    @inbounds propagate(s, p, l)
    acc_rate += local_updates(s, p, l)
  end

  if mod(i, 10) == 0
    acc_rate = acc_rate / (10 * 2 * p.slices)
    println("\t", i)
    @printf("\t\telapsed time: %.2fs\n", toq())
    @printf("\t\tacceptance rate: %.1f%%\n", acc_rate*100)

    # adaption (first half of thermalization)
    if i < p.thermalization / 2 + 1
      if acc_rate < 0.5
        @printf("\t\tshrinking box: %.2f\n", 0.9*p.box.b)
        p.box = Uniform(-0.9*p.box.b,0.9*p.box.b)
      else
        @printf("\t\tenlarging box: %.2f\n", 1.1*p.box.b)
        p.box = Uniform(-1.1*p.box.b,1.1*p.box.b)
      end
    end
    acc_rate = 0.0
    tic()
  end

end
toq();

println("")
initialize_stack(s, p, l)
@time build_stack(s, p, l)
println("Propagating", s.current_slice, " ", s.direction)
propagate(s, p, l)
println("")


println("\nMeasurements - ", p.measurements)
cs = min(p.measurements, 128)

configurations = Observable{Float64}("configurations", size(p.hsfield), cs)
greens = Observable{Complex{Float64}}("greens", size(s.greens), cs)

boson_action = Observable{Float64}("boson action", cs)
mean_abs_op = Observable{Float64}("mean abs op", cs)
mean_op = Observable{Float64}("mean op", (3), cs)

acc_rate = 0.0
tic()
for i in 1:p.measurements
  for u in 1:2 * p.slices
    @inbounds propagate(s, p, l)
    acc_rate += local_updates(s, p, l)

    if s.current_slice == 200 && s.direction == 1 # measure criterium
      # println("\t\tMeasuring")
      add_element(boson_action, p.boson_action)

      curr_mean_abs_op, curr_mean_op = measure_op(s,p,l)
      add_element(mean_abs_op, curr_mean_abs_op)
      add_element(mean_op, curr_mean_op)

      add_element(configurations, p.hsfield)
      add_element(greens, s.greens)

      if mod(i, cs) == 0
          println("Dumping...")
        @time begin
          obs2hdf5(output_file, configurations)
          obs2hdf5(output_file, greens)

          obs2hdf5(output_file, boson_action)
          obs2hdf5(output_file, mean_abs_op)
          obs2hdf5(output_file, mean_op)
          clear(boson_action)
          clear(mean_abs_op)
          clear(mean_op)

          clear(configurations)
          clear(greens)
          println("Dumping block of $cs datapoints was a success")
        end
      end
    end
  end
  if mod(i, 10) == 0
    acc_rate = acc_rate / (10 * 2 * p.slices)
    println("\t", i)
    @printf("\t\telapsed time: %.2fs\n", toq())
    @printf("\t\tacceptance rate: %.1f%%\n", acc_rate*100)
    acc_rate = 0.0
    tic()
  end
end
toq();

# end # inbounds
