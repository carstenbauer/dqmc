julia_code_file = "/projects/ag-trebst/bauer/codes/julia-sdw-dqmc/dqmc.jl"
# output_root = "/projects/ag-trebst/bauer/sdwO3/julia-dqmc"
output_root = pwd()
lattice_dir = "/projects/ag-trebst/bauer/lattices"

include("$(dirname(julia_code_file))/xml_parameters.jl")
using Git
cd(dirname(julia_code_file))
# if Git.unstaged() || Git.staged() error("GIT: Code has staged or unstaged changes. Commit before running simulations.") end
commit = Git.head()

if !isdir(output_root) mkdir(output_root) end
cd(output_root)



const R_POINTS = [0.0,0.6,1.2,1.8,2.4,3.0,3.6,4.2,4.8,5.4,6.0]
const OPDIM = 2

for L in [8]
for beta in [5, 10]
for dt in [0.1]
# for (k, seed) in enumerate([55796])
W = L # square lattice

dir = "L_$(L)"
if !isdir(dir) mkdir(dir) end
cd(dir)
prefix = "sdwO$(OPDIM)_L_$(L)_B_$(beta)_dt_$(dt)_$(k)"

mkdir(prefix)
cd(prefix)

p = Dict{Any, Any}("LATTICE_FILE"=>["$lattice_dir/square_L_$(L)_W_$(W).xml"], "SLICES"=>[Int(beta / dt)], "DELTA_TAU"=>[dt], "SAFE_MULT"=>[10], "C"=>[3.0], "GLOBAL_UPDATES"=>["TRUE"], "GLOBAL_RATE"=>[5], "U"=>[1.0], "R"=>R_POINTS, "LAMBDA"=>[2.0], "HOPPINGS"=>["1.0,0.5,-0.5,-1.0"], "MU"=>[-0.5])
p["THERMALIZATION"] = 1000
p["MEASUREMENTS"] = 20000
p["WRITE_EVERY_NTH"] = 10
p["BFIELD"] = true
p["CHECKERBOARD"] = true
p["OPDIM"] = OPDIM

p["GIT_COMMIT"] = [commit]
paramset2xml(p, prefix)




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
source /projects/ag-trebst/bauer/julia-mkl-intel-env
cd $(output_root)/$(dir)/$(prefix)/
julia -O3 $(julia_code_file) $(prefix) \$\{SLURM_ARRAY_TASK_ID\}
"""
f = open("$(prefix).sh", "w")
write(f, job_cheops)
close(f)

cd("../..")
# end # seed loop
end
end
end
