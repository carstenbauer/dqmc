julia_code_file = "/projects/ag-trebst/bauer/codes/julia-sdw-dqmc/dqmc.jl"
output_root = "/projects/ag-trebst/bauer/sdwO3/julia-dqmc"
lattice_dir = "/projects/ag-trebst/bauer/lattices"

include("$(dirname(julia_code_file))/xml_parameters.jl")

if !isdir(output_root) mkdir(output_root) end
cd(output_root)

for L in [6]
for beta in [10]
for dt in [0.1]
for (k, seed) in enumerate([55796])
W = L # square lattice

dir = "L_$(L)"
if !isdir(dir) mkdir(dir) end
cd(dir)
prefix = "sdwO3_L_$(L)_B_$(beta)_dt_$(dt)_$(k)"

mkdir(prefix)
cd(prefix)

p = Dict{Any, Any}("LATTICE_FILE"=>["$lattice_dir/square_L_$(L)_W_$(W).xml"], "SLICES"=>[Int(beta / dt)], "DELTA_TAU"=>[dt], "SAFE_MULT"=>[10], "U"=>[1.0], "R"=>[8.0, 9.0, 10.0], "LAMBDA"=>[3., 4., 5.], "HOPPINGS"=>["1.0,0.5,0.5,1.0"], "MU"=>[0.5], "SEED"=>[55796], "BOX_HALF_LENGTH"=>[0.2])
p["THERMALIZATION"] = 50
p["MEASUREMENTS"] = 50
parameterset2xml(p, prefix)


job_cheops = """
    #!/bin/bash -l
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --mem=1gb
    #SBATCH --time=24:00:00
    #SBATCH --account=UniKoeln

    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    source /projects/ag-trebst/julia-0.5/julia_env
    cd $(output_root)/$(dir)/$(prefix)/
    julia $(julia_code_file) $(prefix) \$\{SLURM_ARRAY_TASK_ID\}
    """
f = open("$(prefix).sh", "w")
write(f, job_cheops)
close(f)

cd("../..")
end
end
end
end
