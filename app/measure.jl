# -------------------------------------------------------
#            Check input arguments
# -------------------------------------------------------
length(ARGS) >= 1 || error("input arg: prefix.meas.xml [or prefix.in.xml]")

# support for symbol link to app/measure.jl in root of repo
__FILE__ = joinpath(dirname(@__FILE__), splitpath(dirname(@__FILE__))[end] == "app" ? "" : "app")

# -------------------------------------------------------
#            Start + Check git branch
# -------------------------------------------------------
using Dates, Printf, Pkg
const start_time = now()
println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))
println("Hostname: ", gethostname())



function is_dqmc_env_activated()
    project_file = Base.active_project()
    project = Pkg.Types.read_project(project_file)
    return !isnothing(project.name) && lowercase(project.name) == "dqmc"
end

if !is_dqmc_env_activated()
  println("Activating DQMC environment.")
  haskey(ENV, "JULIA_DQMC") || error("DQMC environment not loaded and JULIA_DQMC env variable not set!")
  Pkg.activate(ENV["JULIA_DQMC"])
  is_dqmc_env_activated() || error("Invalid JULIA_DQMC env variable value.")
end



using Git
branch = Git.branch(dir=dirname(@__FILE__)).string[1:end-1]
if branch != "master"
  println("!!!Not on branch master but \"$(branch)\"!!!")
  flush(stdout)
end



# -------------------------------------------------------
#                   Load Framework
# -------------------------------------------------------
include(joinpath(__FILE__, "../src/measure_framework.jl"))

if HDF5.libversion.minor |> Int < 10
  @warn "You are using a HDF5 version < 1.10 in which space isn't freed automatically.
          Files might become very large, in particular for small values of SAVE_AFTER!"
end



# #######################################################
#                    Start script
# #######################################################
# -------------------------------------------------------
#           Parse ARGS and maybe .meas.xml
# -------------------------------------------------------

const arg = ARGS[1]

isfile(arg) || error("Input file $arg not found.")
if endswith(arg, ".meas.xml")
  mp = measxml2MeasParams(arg)
elseif endswith(arg, ".in.xml")
  measxml = arg[1:end-7]*".meas.xml"
  isfile(measxml) || error("File $(measxml) not found. Input arg was: $(arg).")
  mp = measxml2MeasParams(measxml)
else
  error("No .meas.xml or .in.xml file found. Input arg was: $(arg)")
end



# set num threads
try
  BLAS.set_num_threads(mp.num_threads)
  FFTW.set_num_threads(mp.num_threads)
  ENV["OMP_NUM_THREADS"] = mp.num_threads
  ENV["MKL_NUM_THREADS"] = mp.num_threads
  ENV["JULIA_NUM_THREADS"] = 1
catch
end


# -------------------------------------------------------
#                     Let's go
# -------------------------------------------------------
main(mp)


# -------------------------------------------------------
#                       Exit
# -------------------------------------------------------
end_time = now()
println("\nEnded: ", Dates.format(end_time, "d.u yyyy HH:MM"))
@printf("Duration: %.2f minutes", (end_time - start_time).value / 1000 / 60)