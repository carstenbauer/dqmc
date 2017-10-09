julia_code_file = "/projects/ag-trebst/bauer/codes/julia-sdw-dqmc/dqmc.jl"
output_root = "/projects/ag-trebst/bauer/sdwO3/julia-dqmc"
lattice_dir = "/projects/ag-trebst/bauer/lattices"

include("$(dirname(julia_code_file))/xml_parameters.jl")
using Git
cd(dirname(julia_code_file))
# if Git.unstaged() || Git.staged() error("GIT: Code has staged or unstaged changes. Commit before running simulations.") end
commit = Git.head()

if !isdir(output_root) mkdir(output_root) end
cd(output_root)

for L in [8]
for beta in [5, 10]
for dt in [0.1]
for (k, seed) in enumerate([55796])
W = L # square lattice

dir = "L_$(L)"
if !isdir(dir) mkdir(dir) end
cd(dir)
prefix = "sdwO3_L_$(L)_B_$(beta)_dt_$(dt)_$(k)"

mkdir(prefix)
cd(prefix)

p = Dict{Any, Any}("LATTICE_FILE"=>["$lattice_dir/square_L_$(L)_W_$(W).xml"], "SLICES"=>[Int(beta / dt)], "DELTA_TAU"=>[dt], "SAFE_MULT"=>[10], "C"=>[3.0], "GLOBAL_UPDATES"=>["TRUE"], "GLOBAL_RATE"=>[10], "U"=>[1.0], "R"=>[0.0,0.6,1.2,1.8,2.4,3.0,3.6,4.2,4.8,5.4,6.0], "LAMBDA"=>[2.0], "HOPPINGS"=>["1.0,0.5,-0.5,-1.0"], "MU"=>[-0.5], "SEED"=>[55796], "BOX_HALF_LENGTH"=>[0.1])
p["THERMALIZATION"] = 1000
p["MEASUREMENTS"] = 20000
p["WRITE_EVERY_NTH"] = 10
p["B_FIELD"] = false

p["GIT_COMMIT"] = [commit]
parameterset2xml(p, prefix)

#SBATCH --cpus-per-task=4
job_cheops = """
#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4gb
#SBATCH --time=7-00:00:00
#SBATCH --output=$(prefix).task%a.out.log
#SBATCH --job-name=$(prefix)

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JULIA_PKGDIR=/projects/ag-trebst/bauer/.julia
source /projects/ag-trebst/bauer/julia-curr-env
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
