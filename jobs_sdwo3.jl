# copied from peter

import HDF5
include("la.jl")
include("parameters.jl")
include("lattice.jl")
include("interactions.jl")
include("stack.jl")
include("updates.jl")
include("xml_parameters.jl")
include("observable.jl")

for L in [6]
# L = 3
W = L
theta = 10
dt = 0.05

lattice_dir = "/projects/ag-trebst/bauer/lattices"
dir = "L_$(L)_W_$(W)"
cd(dir)
prefix = "honeycomb_L_$(L)_W_$(W)_theta_$(theta)_dt_$(dt)"

mkdir(prefix)
cd(prefix)

p = Dict{Any, Any}("LATTICE_FILE"=>["$lattice_dir/honeycomb_L_$(L)_W_$(W).xml"], "SLICES"=>[Int(theta / dt)], "DELTA_TAU"=>[dt], "SAFE_MULT"=>[10], "U"=>[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], "HOPPINGS"=>["1.0,1.0,1.0"], "LATTICE"=>["honeycomb"], "SEED"=>[13])
p["THERMALIZATION"] = 1024
p["MEASUREMENTS"] = 8192
parameters2xml(p, prefix)


job_cheops = """
    #!/bin/bash -l
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --mem=1gb
    #SBATCH --time=24:00:00
    #SBATCH --account=UniKoeln

    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    source ~/.bashrc
    cd /projects/ag-trebst/pboertz/julia_dqmc/honeycomb/$(dir)/$(prefix)/
    julia /projects/ag-trebst/pboertz/codes/julia_dqmc/src/dqmc.jl $(prefix) \$\{SLURM_ARRAY_TASK_ID\}
    """
f = open("$(prefix).sh", "w")
write(f, job_cheops)
close(f)

cd("../..")
end
