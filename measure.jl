# measure.jl called with arguments: inputfile.out.h5
# expects additional file inputfile.measure.in.xml
start_time = now()
println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))

using Helpers
using Git
# include("linalg.jl")
include("parameters.jl")
include("xml_parameters.jl")
# include("lattice.jl")
# include("checkerboard.jl")
# include("interactions.jl")
# include("action.jl")
# include("stack.jl")
# include("local_updates.jl")
# include("global_updates.jl")
# include("observable.jl")
# include("boson_measurements.jl")
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
if HDF5.has(f, "parameters/TEST") HDF5.o_delete(f["parameters"],"TEST") end
f["parameters/TEST"] = 43

# Check and store code version (git commit)
if Git.head(dir=dirname(@__FILE__)) != f["parameters/GIT_COMMIT"]
  warn("Git commit in h5 file does not match current commit of code.")
end

p = Parameters()
p.thermalization = parse(Int64, read(f["parameters/THERMALIZATION"]))
p.measurements = parse(Int64, read(f["parameters/MEASUREMENTS"]))
p.slices = parse(Int64, read(f["parameters/SLICES"]))
p.delta_tau = parse(Float64, read(f["parameters/DELTA_TAU"]))
p.safe_mult = parse(Int64, read(f["parameters/SAFE_MULT"]))
p.lattice_file = read(f["parameters/LATTICE_FILE"])
p.mu = parse(Float64, read(f["parameters/MU"]))
p.lambda = parse(Float64, read(f["parameters/LAMBDA"]))
p.r = parse(Float64, read(f["parameters/R"]))
p.c = parse(Float64, read(f["parameters/C"]))
p.u = parse(Float64, read(f["parameters/U"]))
p.global_updates = parse(Bool, lowercase(read(f["parameters/GLOBAL_UPDATES"])))
p.beta = p.slices * p.delta_tau
p.flv = 4
p.box = Uniform(-parse(Float64, read(f["parameters/BOX_HALF_LENGTH"])),parse(Float64, read(f["parameters/BOX_HALF_LENGTH"])))
p.box_global = Uniform(-parse(Float64, read(f["parameters/BOX_GLOBAL_HALF_LENGTH"])),parse(Float64, read(f["parameters/BOX_GLOBAL_HALF_LENGTH"])))
p.global_rate = parse(Int64, read(f["parameters/GLOBAL_RATE"]))

close(f)



end_time = now()
println("Ended: ", Dates.format(end_time, "d.u yyyy HH:MM"))
@printf("Duration: %.2f minutes", (end_time - start_time).value/1000./60.)
