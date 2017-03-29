# measure.jl called with arguments: sdwO3_L_4_B_2_dt_0.1_1 ${SLURM_ARRAY_TASK_ID}
start_time = now()
println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))

using Helpers
using Git
# include("linalg.jl")
# include("parameters.jl")
# include("xml_parameters.jl")
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

ARGS = ["sdwO3_L_4_B_2_dt_0.1_1", 1]
prefix = convert(String, ARGS[1])
idx = 1
try
  idx = parse(Int, ARGS[2]) # SLURM_ARRAY_TASK_ID
end
input_file = prefix * ".task" * string(idx) * ".out.h5"



end_time = now()
println("Ended: ", Dates.format(end_time, "d.u yyyy HH:MM"))
@printf("Duration: %.2f minutes", (end_time - start_time).value/1000./60.)
