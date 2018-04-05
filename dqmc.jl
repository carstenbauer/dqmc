# dqmc.jl called with arguments: whatever.in.xml
start_time = now()
println("Started: ", Dates.format(start_time, "d.u yyyy HH:MM"))
println("Hostname: ", gethostname())

using Helpers
using Git
include("parameters.jl")

### PROGRAM ARGUMENTS
if length(ARGS) == 1
  # ARGS = ["whatever.in.xml"]
  input_xml = ARGS[1]
  output_file = input_xml[1:searchindex(input_xml, ".in.xml")-1]*".out.h5.running"
elseif length(ARGS) == 2
  # Backward compatibility
  # ARGS = ["sdwO3_L_4_B_2_dt_0.1_2", 1]
  prefix = convert(String, ARGS[1])
  idx = 1
  try idx = parse(Int, ARGS[2]); end # SLURM_ARRAY_TASK_ID 
  output_file = prefix * ".task" * string(idx) * ".out.h5.running"

  println("Prefix is ", prefix, " and idx is ", idx)
  input_xml = prefix * ".task" * string(idx) * ".in.xml"
else
  error("Call with \"whatever.in.xml\" or e.g. \"sdwO3_L_4_B_2_dt_0.1_1 \${SLURM_ARRAY_TASK_ID}\"")
end

# hdf5 write test/ dump git commit
branch = Git.branch(dir=dirname(@__FILE__)).string[1:end-1]
if branch != "master"
  warn("Not on branch master but \"$(branch)\"!!!")
  flush(STDOUT)
end

### PARAMETERS
p = Parameters()
p.output_file = output_file
xml2parameters!(p, input_xml)

# mv old .out.h5 to .out.h5.running (and then try to resume below)
(isfile(output_file[1:end-8]) && !isfile(output_file)) && mv(output_file[1:end-8], output_file)

# check if there is a resumable running file
if isfile(output_file)
  try
    h5open(output_file) do f
      if HDF5.has(f, "resume") && read(f["count"]) > 0 && read(f["GIT_BRANCH_DQMC"]) == branch
        p.resume = true
      end
    end
  end
end
if !p.resume
  # overwrite (potential) running file
  h5open(output_file, "w") do f
    f["GIT_COMMIT_DQMC"] = Git.head(dir=dirname(@__FILE__)).string
    f["GIT_BRANCH_DQMC"] = branch
    f["RUNNING"] = 1
  end
end

if !p.resume
  parameters2hdf5(p, p.output_file)
else
  println()
  println("RESUMING MODE -----------------")
  h5open(output_file, "r+") do f
    !HDF5.has(f, "RUNNING") || HDF5.o_delete(f, "RUNNING")
    f["RUNNING"] = 1
    !HDF5.has(f, "RESUME") || HDF5.o_delete(f, "RESUME")
    f["RESUME"] = 1
  end

  global const prevcount = h5read(output_file, "count")
  println("Found $(prevcount) configurations.")
  println()
  lastmeasurements = h5read(output_file, "params/measurements")
  global const prevmeasurements = prevcount * p.write_every_nth
  if prevmeasurements < lastmeasurements
    println("Finishing last run (i.e. overriding p.measurements)")
    p.measurements = lastmeasurements - prevmeasurements
  end
  global const lastconf = squeeze(h5read(output_file, "configurations", (:,:,:,prevcount)), 4)
  global const lastgreens = squeeze(h5read(output_file, "obs/greens/timeseries_real", (:,:,prevcount)) + 
                              im*h5read(output_file, "obs/greens/timeseries_imag", (:,:,prevcount)), 3)
end

println("HoppingType = ", HoppingType)
println("GreensType = ", GreensType)



include("lattice.jl")
include("stack.jl")

mutable struct Analysis
    acc_rate::Float64
    acc_rate_global::Float64
    prop_global::Int
    acc_global::Int

    Analysis() = new()
end

abstract type Checkerboard end
abstract type CBTrue <: Checkerboard end
abstract type CBFalse <: Checkerboard end
abstract type CBGeneric <: CBTrue end
abstract type CBAssaad <: CBTrue end

mutable struct DQMC{C<:Checkerboard}
  p::Parameters
  l::Lattice
  s::Stack
  a::Analysis
end

DQMC(p::Parameters) = begin
  CB = CBFalse
  p.chkr && (CB = iseven(p.L) ? CBGeneric : CBAssaad)
  mc = DQMC{CB}(p, Lattice(), Stack(), Analysis())
  load_lattice(mc)
  mc
end

# cosmetics
import Base.summary
import Base.show
Base.summary(mc::DQMC) = "DQMC"
function Base.show(io::IO, mc::DQMC{C}) where C<:Checkerboard
    print(io, "DQMC of O($(mc.p.opdim)) model\n")
    print(io, "r = ", mc.p.r, ", λ = ", mc.p.lambda, ", c = ", mc.p.c, ", u = ", mc.p.u, "\n")
    print(io, "Beta: ", mc.p.beta, " (T ≈ $(round(1/mc.p.beta, 3)))", "\n")
    print(io, "Checkerboard: ", C, "\n")
    print(io, "B-field: ", mc.p.Bfield)
end
Base.show(io::IO, m::MIME"text/plain", mc::DQMC) = print(io, mc)

include("linalg.jl")
include("hoppings.jl")
if iseven(p.L)
  include("checkerboard.jl")
else
  include("checkerboard_generic.jl")
end
include("interactions.jl")
include("action.jl")
include("local_updates.jl")
include("global_updates.jl")
include("observable.jl")
include("boson_measurements.jl")
include("fermion_measurements.jl")

function init!(mc::DQMC)
    srand(mc.p.seed); # init RNG

    # Init hsfield
    println("\nInitializing HS field")
    mc.p.hsfield = rand(mc.p.opdim,mc.l.sites,mc.p.slices)
    println("Initializing boson action\n")
    mc.p.boson_action = calculate_boson_action(mc)

    global const eye_flv = eye(mc.p.flv,mc.p.flv)
    global const eye_full = eye(mc.p.flv*mc.l.sites,mc.p.flv*mc.l.sites)
    global const ones_vec = ones(mc.p.flv*mc.l.sites)

    # stack init and test
    initialize_stack(mc)
    println("Building stack")
    build_stack(mc)
    println("Initial propagate: ", mc.s.current_slice, " ", mc.s.direction)
    propagate(mc)
end

function run!(mc::DQMC)
    println("\n\nMC Thermalize - ", mc.p.thermalization)
    flush(STDOUT)
    thermalize!(mc)

    h5open(mc.p.output_file, "r+") do f
      write(f, "resume/box", mc.p.box.b)
      write(f, "resume/box_global", mc.p.box_global.b)
    end

    println("\n\nMC Measure - ", mc.p.measurements)
    flush(STDOUT)
    measure!(mc)
    nothing
end


function resume!(mc::DQMC)
    # Init hsfield
    println("\nLoading last HS field")
    p.hsfield = deepcopy(lastconf)
    println("Initializing boson action\n")
    p.boson_action = calculate_boson_action(mc)

    global const eye_flv = eye(p.flv,p.flv)
    global const eye_full = eye(p.flv*l.sites,p.flv*l.sites)
    global const ones_vec = ones(p.flv*l.sites)

    h5open(p.output_file, "r") do f
      box = read(f, "resume/box")
      box_global = read(f, "resume/box_global")
      p.box = Uniform(-box, box)
      p.box_global = Uniform(-box_global, box_global)
    end

    ### MONTE CARLO
    println("\n\nMC Measure (resuming) - ", p.measurements, " (total $(p.measurements + prevmeasurements))")
    flush(STDOUT)
    measure!(mc)

    nothing
end


function thermalize!(mc::DQMC)
    const a = mc.a
    const p = mc.p

    a.acc_rate = 0.0
    a.acc_rate_global = 0.0
    a.prop_global = 0
    a.acc_global = 0
    tic()
    for i in 1:p.thermalization
      for u in 1:2 * p.slices
        update(mc, i)
      end

      if mod(i, 10) == 0
        a.acc_rate = a.acc_rate / (10 * 2 * p.slices)
        a.acc_rate_global = a.acc_rate_global / (10 / p.global_rate)
        println("\t", i)
        @printf("\t\tup-down sweep dur: %.2fs\n", toq()/10)
        @printf("\t\tacc rate (local) : %.1f%%\n", a.acc_rate*100)
        if p.global_updates
          @printf("\t\tacc rate (global): %.1f%%\n", a.acc_rate_global*100)
          @printf("\t\tacc rate (global, overall): %.1f%%\n", a.acc_global/a.prop_global*100)
        end

        # adaption (first half of thermalization)
        if i < p.thermalization / 2 + 1
          if a.acc_rate < 0.5
            @printf("\t\tshrinking box: %.2f\n", 0.9*p.box.b)
            p.box = Uniform(-0.9*p.box.b,0.9*p.box.b)
          else
            @printf("\t\tenlarging box: %.2f\n", 1.1*p.box.b)
            p.box = Uniform(-1.1*p.box.b,1.1*p.box.b)
          end

          if p.global_updates
          if a.acc_global/a.prop_global < 0.5
            @printf("\t\tshrinking box_global: %.2f\n", 0.9*p.box_global.b)
            p.box_global = Uniform(-0.9*p.box_global.b,0.9*p.box_global.b)
          else
            @printf("\t\tenlarging box_global: %.2f\n", 1.1*p.box_global.b)
            p.box_global = Uniform(-1.1*p.box_global.b,1.1*p.box_global.b)
          end
          end
        end
        a.acc_rate = 0.0
        a.acc_rate_global = 0.0
        flush(STDOUT)
        tic()
      end

    end
    toq();
    nothing
end


function measure!(mc::DQMC)
    const a = mc.a
    const l = mc.l
    const p = mc.p
    const s = mc.s

    initialize_stack(mc)
    println("Renewing stack")
    build_stack(mc)
    println("Initial propagate: ", s.current_slice, " ", s.direction)
    propagate(mc)

    cs = min(p.measurements, 100)

    configurations = Observable{Float64}("configurations", size(p.hsfield), cs)
    greens = Observable{GreensType}("greens", size(s.greens), cs)

    boson_action = Observable{Float64}("boson_action", cs)
    mean_abs_op = Observable{Float64}("mean_abs_op", cs)
    mean_op = Observable{Float64}("mean_op", (p.opdim), cs)

    i_start = 1
    i_end = p.measurements

    if p.resume
      restorerng(p.output_file; group="resume/rng")
      togo = mod1(prevmeasurements, p.write_every_nth)-1
      i_start = prevmeasurements-togo+1
      i_end = p.measurements + prevmeasurements
    end

    acc_rate = 0.0
    acc_rate_global = 0.0
    tic()
    for i in i_start:i_end
      for u in 1:2 * p.slices
        update(mc, i)

        if s.current_slice == p.slices && s.direction == -1 && (i-1)%p.write_every_nth == 0 # measure criterium
          # println("\t\tMeasuring")
          add_element(boson_action, p.boson_action)

          curr_mean_abs_op, curr_mean_op = measure_op(p.hsfield)
          add_element(mean_abs_op, curr_mean_abs_op)
          add_element(mean_op, curr_mean_op)

          add_element(configurations, p.hsfield)
          add_element(greens, s.greens)
          
          if p.chkr
            effective_greens2greens!(p, l, greens.timeseries[:,:,greens.count])
          else
            effective_greens2greens_no_chkr!(p, l, greens.timeseries[:,:,greens.count])
          end

          # compare(greens.timeseries[:,:,greens.count], measure_greens_and_logdet(s, p, l, p.safe_mult)[1])

          # add_element(greens, measure_greens_and_logdet(s, p, l, p.safe_mult)[1])

          if boson_action.count == cs
              println("Dumping...")
            @time begin
              h5open(p.output_file, "r+") do f
                confs2hdf5(f, configurations)
                obs2hdf5(f, greens)

                obs2hdf5(f, boson_action)
                obs2hdf5(f, mean_abs_op)
                obs2hdf5(f, mean_op)
              end
              clear(boson_action)
              clear(mean_abs_op)
              clear(mean_op)

              clear(configurations)
              clear(greens)

              saverng(p.output_file; group="resume/rng")
              println("Dumping block of $cs datapoints was a success")
              flush(STDOUT)
            end
          end
        end
      end
      if mod(i, 100) == 0
        a.acc_rate = a.acc_rate / (100 * 2 * p.slices)
        a.acc_rate_global = a.acc_rate_global / (100 / p.global_rate)
        println("\t", i)
        @printf("\t\tup-down sweep dur: %.2fs\n", toq()/100)
        @printf("\t\tacc rate (local) : %.1f%%\n", a.acc_rate*100)
        if p.global_updates
          @printf("\t\tacc rate (global): %.1f%%\n", a.acc_rate_global*100)
          @printf("\t\tacc rate (global, overall): %.1f%%\n", a.acc_global/a.prop_global*100)
        end
        a.acc_rate = 0.0
        a.acc_rate_global = 0.0
        flush(STDOUT)
        tic()
      end
    end
    toq();
    nothing
end

function update(mc::DQMC, i::Int)
    const p = mc.p
    const s = mc.s
    const l = mc.l
    const a = mc.a

    propagate(s, p, l)

    if p.global_updates && (s.current_slice == p.slices && s.direction == -1 && mod(i, p.global_rate) == 0)
      a.prop_global += 1
      b = global_update(s, p, l)
      a.acc_rate_global += b
      a.acc_global += b
    end

    a.acc_rate += local_updates(s, p, l)
    nothing
end

# main

mc = DQMC(p)

@printf("It took %.2f minutes to prepare everything. \n", (now() - start_time).value/1000./60.)

if !mc.p.resume
  init!(mc)
  run!(mc)
else
  resume!(mc)
end

h5open(output_file, "r+") do f
  HDF5.o_delete(f, "RUNNING")
  f["RUNNING"] = 0
end
mv(output_file, output_file[1:end-8], remove_destination=true) # .out.h5.running -> .out.h5

end_time = now()
println("Ended: ", Dates.format(end_time, "d.u yyyy HH:MM"))
@printf("Duration: %.2f minutes", (end_time - start_time).value/1000./60.)