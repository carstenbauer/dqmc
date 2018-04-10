# measure.jl called with argument: inputfile.out.h5 OR inputfile.measure.in.xml
# expects additional file inputfile.measure.in.xml OR inputfile.out.h5
start_time = now()
println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))

using Helpers
using Git
include("parameters.jl")
include("observable.jl")

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
git_commit = Git.head(dir=dirname(@__FILE__))
if !HDF5.exists(f, "GIT_COMMIT_DQMC") || (git_commit != f["GIT_COMMIT_DQMC"])
  warn("Git commit used for dqmc doesn't match current commit of code!!!")
end
if Git.branch(dir=dirname(@__FILE__)).string[1:end-1] != "master"
  warn("Not on branch master!!!")
end
if HDF5.exists(f, "GIT_COMMIT_MEASURE")
  warn("Overwriting GIT_COMMIT_MEASURE!")
  HDF5.o_delete(f, "GIT_COMMIT_MEASURE")
end
f["GIT_COMMIT_MEASURE"] = git_commit.string

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
  params = xml2params(input_file)

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


# load configurations
confs = read(f["configurations"])
num_confs = size(confs)[end]
println("\n Found $(num_confs) configurations.")

close(f)


######### All set. Let's get going ##########

p = Params()
#hdf52parameters!_old(p, output_file)
hdf52parameters!(p, output_file)

include("lattice.jl")
include("stack.jl")
include("linalg.jl")
include("hoppings.jl")
include("checkerboard.jl")
include("interactions.jl")
include("action.jl")
# include("local_updates.jl")
# include("global_updates.jl")
include("boson_measurements.jl")
include("fermion_measurements.jl")

l = Lattice()
load_lattice(p,l)

s = Stack() # only used for temporary arrays, e.g. s.eV
initialize_stack(s, p, l)

println("")
dump(m)

function Measure(m::Measurements, s::Stack, p::Params, l::Lattice, confs::Array{Float64, 4})

  m.safe_mult = (m.safe_mult == -1) ? p.safe_mult : m.safe_mult

  tic()
  println("\nMeasuring...")
  cs = num_confs # currently, we do not dump intermediate results

  chi = Observable{Float64}("boson_suscept", cs)
  m2 = Observable{Float64}("m2", cs)
  m4 = Observable{Float64}("m4", cs)

  greens = Observable{GreensType}("greens", (p.flv*l.sites, p.flv*l.sites), cs)
  fermion_action = Observable{Float64}("fermion_action", cs)
  action = Observable{Float64}("action", cs)

  S_b = hdf52obs(output_file, "boson_action").timeseries

  elapsed_time = 0.0
  for c in 1:num_confs
    conf = confs[:,:,:,c]

    if m.binder
      curr_m2, curr_m4 = measure_binder_factors(conf)
      add_element(m2, curr_m2)
      add_element(m4, curr_m4)
    end

    if m.boson_suscept
      add_element(chi, measure_chi_static(conf))
    end

    if m.greens
      p.hsfield = conf
      curr_greens, ld = measure_greens_and_logdet(s, p, l, m.safe_mult)
      add_element(greens, curr_greens)

      S_f = ld # O(3) default
      if p.opdim == 2 || p.opdim == 1
        # O(1) & O(2)
        # S_f = log(det(G)^2) = 2*log(det(G)) = 2*ld
        S_f = 2*ld
      end

      add_element(fermion_action, S_f)
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
end

Measure(m,s,p,l,confs)

end_time = now()
println("\nEnded: ", Dates.format(end_time, "d.u yyyy HH:MM"))
@printf("Duration: %.2f minutes", (end_time - start_time).value/1000./60.)
