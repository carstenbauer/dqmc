# -------------------------------------------------------
#                    Abstract types
# -------------------------------------------------------
abstract type Checkerboard end
abstract type CBTrue <: Checkerboard end
abstract type CBFalse <: Checkerboard end
abstract type CBGeneric <: CBTrue end
abstract type CBAssaad <: CBTrue end

abstract type AbstractDQMC{C<:Checkerboard, GreensEltype<:Number, HoppingEltype<:Number} end

(@isdefined DQMC_CBTrue) || (global const DQMC_CBTrue = AbstractDQMC{C} where C<:CBTrue)
(@isdefined DQMC_CBFalse) || (global const DQMC_CBFalse = AbstractDQMC{C} where C<:CBFalse)



# -------------------------------------------------------
#                    Includes
# -------------------------------------------------------
using Helpers
using MonteCarloObservable
using TimerOutputs
using FFTW
using HDF5
using JLD
using LightXML

# using Iterators
using Dates
using LinearAlgebra
using SparseArrays
using Statistics
using Printf
using Random


(@isdefined TIMING) || (global const TIMING = false)
macro mytimeit(exprs...)
    if TIMING
        return :(@timeit($(esc.(exprs)...)))
    else
        return esc(exprs[end])
    end
end

include("parameters.jl")
include("lattice.jl")
include("stack.jl")
include("slice_matrices.jl")
include("linalg.jl")
include("hoppings.jl")
include("hoppings_checkerboard.jl")
include("hoppings_checkerboard_generic.jl")
include("interactions.jl")
include("action.jl")
include("local_updates.jl")
include("global_updates.jl")
# include("observable.jl")
include("boson_measurements.jl")
include("fermion_measurements.jl")

# tools for apps
include("tools/xml_parameters.jl")
include("tools/hdf5_parameters.jl")



# -------------------------------------------------------
#                    Concrete types
# -------------------------------------------------------
mutable struct Analysis
  acc_rate::Float64
  acc_rate_global::Float64
  prop_global::Int
  acc_global::Int
  to::TimerOutputs.TimerOutput

  Analysis() = new(0.0,0.0,0,0,TimerOutput())
end

mutable struct DQMC{C<:Checkerboard, GreensEltype<:Number, HoppingEltype<:Number} <: AbstractDQMC{C, GreensEltype, HoppingEltype}
  p::Params
  l::Lattice{HoppingEltype}
  s::Stack{GreensEltype}
  a::Analysis
end

DQMC(p::Params) = begin
  CB = CBFalse

  # choose checkerboard variant
  if p.chkr
    if p.Nhoppings == "none" && p.NNhoppings == "none"
      CB = iseven(p.L) ? CBAssaad : CBGeneric
    else
      CB = CBGeneric
    end
  end
  println()
  @show CB
  @show p.hoppings
  @show p.Nhoppings
  @show p.NNhoppings
  @show p.lattice_file
  println()

  ### SET DATATYPES
  G = ComplexF64
  H = ComplexF64
  if !p.Bfield
    H = Float64;
    G = p.opdim > 1 ? ComplexF64 : Float64; # O(1) -> real GF
  end

  mc = DQMC{CB,G,H}(p, Lattice{H}(), Stack{G}(), Analysis())
  load_lattice(mc)
  mc
end

# type helpers
@inline geltype(mc::DQMC{CB, G, H}) where {CB, G, H} = G
@inline heltype(mc::DQMC{CB, G, H}) where {CB, G, H} = H
@inline cbtype(mc::DQMC{CB, G, H}) where {CB, G, H} = CB



# -------------------------------------------------------
#                  Monte Carlo
# -------------------------------------------------------
function init!(mc::DQMC)
  Random.seed!(mc.p.seed); # init RNG
  init!(mc, rand(mc.p.opdim,mc.l.sites,mc.p.slices), false)
end
function init!(mc::DQMC, start_conf, init_seed=true)
  a = mc.a
  @mytimeit a.to "init mc" begin
  init_seed && Random.seed!(mc.p.seed); # init RNG

  # Init hsfield
  println("\nInitializing HS field")
  mc.p.hsfield = start_conf
  println("Initializing boson action\n")
  mc.p.boson_action = calc_boson_action(mc)

  # stack init and test
  initialize_stack(mc)
  println("Building stack")
  build_stack(mc)
  println("Initial propagate: ", mc.s.current_slice, " ", mc.s.direction)
  propagate(mc)
  end

  TIMING && show(TimerOutputs.flatten(a.to); allocations = true)
  nothing
end

function run!(mc::DQMC)
  println("\n\nMC Thermalize - ", mc.p.thermalization*2)
  flush(stdout)
  thermalize!(mc)

  h5open(mc.p.output_file, "r+") do f
    HDF5.has(f, "resume/box") && o_delete(f, "resume/box")
    HDF5.has(f, "resume/box_global") && o_delete(f, "resume/box_global")
    write(f, "resume/box", mc.p.box)
    write(f, "resume/box_global", mc.p.box_global)
  end

  println("\n\nMC Measure - ", mc.p.measurements*2)
  flush(stdout)
  measure!(mc)
  nothing
end

function resume!(mc::DQMC, lastconf, prevmeasurements::Int)
  p = mc.p

  # Init hsfield
  println("\nLoading last HS field")
  p.hsfield = copy(lastconf)
  println("Initializing boson action\n")
  p.boson_action = calc_boson_action(mc)

  jldopen(p.output_file, "r") do fjld
    f = fjld.plain
    box = read(f, "resume/box")
    box_global = read(f, "resume/box_global")
    p.box = box
    p.box_global = box_global

    if HDF5.has(f, "thermal_init/udsweep")
      mc.a.to = read(fjld, "thermal_init/udsweep")
    else
      @warn "Resuming measurements but couldn't find \"thermal_init/udsweep\" timer."
    end
  end

  println("\n\nMC Measure (resuming) - ", p.measurements*2, " (total $((p.measurements + prevmeasurements)*2))")
  flush(stdout)
  measure!(mc, prevmeasurements)

  nothing
end











function thermalize!(mc::DQMC)
  a = mc.a
  p = mc.p

  a.acc_rate = 0.0
  a.acc_rate_global = 0.0
  a.prop_global = 0
  a.acc_global = 0

  reset_timer!(a.to)
  if p.prethermalized > 0
    # load old a.to["udsweep"] timer
    jldopen(p.output_file, "r") do f
      a.to = read(f["thermal_init/udsweep"])
    end
  end

  for i in (p.prethermalized+1):p.thermalization
    udswdur = @elapsed @timeit a.to "udsweep" for u in 1:2 * p.slices
      update(mc, i)
    end
    @printf("\tsweep duration: %.4fs\n", udswdur/2)
    flush(stdout)

    if mod(i, 10) == 0
      a.acc_rate = a.acc_rate / (10 * 2 * p.slices)
      a.acc_rate_global = a.acc_rate_global / (10 / p.global_rate)
      println("\n\t", i*2)
      @printf("\t\tsweep dur (total mean): %.4fs\n", TimerOutputs.time(a.to["udsweep"])/2 *10.0^(-9)/TimerOutputs.ncalls(a.to["udsweep"]))
      @printf("\t\tacc rate (local) : %.1f%%\n", a.acc_rate*100)
      if p.global_updates
        @printf("\t\tacc rate (global): %.1f%%\n", a.acc_rate_global*100)
        @printf("\t\tacc rate (global, overall): %.1f%%\n", a.acc_global/a.prop_global*100)
      end

      # adaption (only during thermalization)
      if a.acc_rate < 0.5
        @printf("\t\tshrinking box: %.2f\n", 0.9*p.box)
        p.box = 0.9 * p.box
      else
        @printf("\t\tenlarging box: %.2f\n", 1.1*p.box)
        p.box = 1.1 * p.box
      end

      if p.global_updates
        if a.acc_global/a.prop_global < 0.5
          @printf("\t\tshrinking box_global: %.2f\n", 0.9*p.box_global)
          p.box_global = 0.9 * p.box_global
        else
          @printf("\t\tenlarging box_global: %.2f\n", 1.1*p.box_global)
          p.box_global = 1.1 * p.box_global
        end
      end
      a.acc_rate = 0.0
      a.acc_rate_global = 0.0
      println()
      flush(stdout)
    end

    # Save thermal configuration for "resume"
    if (shutdown = now() >= p.walltimelimit) || i%p.write_every_nth == 0
      _save_thermal(mc, i)
      if shutdown
        println("Approaching wall-time limit. Safely exiting.")
        exit(42)
      end
    end

  end

  if TIMING 
    # display(a.to);
    show(TimerOutputs.flatten(a.to); allocations = true)
    # save(p.output_file[1:end-10]*"timings.jld", "to", a.to); # TODO
    rm(p.output_file)
    exit();
  end
  nothing
end


_save_thermal(mc, i) = begin
    jldopen(mc.p.output_file, "r+") do fjld
      f = fjld.plain
      HDF5.has(f, "thermal_init/conf") && HDF5.o_delete(f, "thermal_init/conf")
      HDF5.has(f, "thermal_init/prethermalized") && HDF5.o_delete(f, "thermal_init/prethermalized")
      HDF5.has(f, "thermal_init/udsweep") && HDF5.o_delete(f, "thermal_init/udsweep")
      write(fjld, "thermal_init/conf", mc.p.hsfield)
      write(fjld, "thermal_init/prethermalized", i)
      write(fjld, "thermal_init/udsweep", mc.a.to["udsweep"])
      (i == mc.p.thermalization) && saverng(f; group="thermal_init/rng") # for future features
    end
end








function measure!(mc::DQMC, prevmeasurements=0)
  a = mc.a
  l = mc.l
  p = mc.p
  s = mc.s

  initialize_stack(mc)
  println("Renewing stack")
  build_stack(mc)
  println("Initial propagate: ", s.current_slice, " ", s.direction)
  propagate(mc)

  cs = choose_chunk_size(mc)

  configurations = Observable(typeof(p.hsfield), "configurations"; alloc=cs, inmemory=false, outfile=p.output_file, group="obs/configurations")
  greens = Observable(typeof(mc.s.greens), "greens"; alloc=cs, inmemory=false, outfile=p.output_file, group="obs/greens")
  occ = Observable(Float64, "occupation"; alloc=cs, inmemory=false, outfile=p.output_file, group="obs/occupation")
  # boson_action = Observable(Float64, "boson_action"; alloc=cs, inmemory=false, outfile=p.output_file, group="obs/boson_action")


  i_start = 1
  i_end = p.measurements

  if p.resume
    restorerng(p.output_file; group="resume/rng")
    togo = mod1(prevmeasurements, p.write_every_nth)-1
    i_start = prevmeasurements-togo+1
    i_end = p.measurements + prevmeasurements
    cs = configurations.alloc
    println("Overriding cs to match cs of previous run (cs = $(cs)).")
  end

  @show cs


  acc_rate = 0.0
  acc_rate_global = 0.0
  for i in i_start:i_end
    @timeit a.to "udsweep" for u in 1:2 * p.slices
      update(mc, i)

      # if s.current_slice == 1 && s.direction == 1 && (i-1)%p.write_every_nth == 0 # measure criterium
      if s.current_slice == p.slices && s.direction == -1 && (i-1)%p.write_every_nth == 0 # measure criterium
        dumping = (length(configurations)+1)%cs == 0
        dumping && println("Dumping... (2*i=$(2*i))")
        # add!(boson_action, p.boson_action)

        add!(configurations, p.hsfield)
        
        # fermionic quantities
        g = wrap_greens(mc,mc.s.greens,mc.s.current_slice,1)
        effective_greens2greens!(mc, g)
        # compare(g, measure_greens(mc))
        add!(greens, g)
        add!(occ, occupation(mc, g))

        dumping && saverng(p.output_file; group="resume/rng")
        dumping && println("Dumping block of $cs datapoints was a success")
        flush(stdout)


        if !dumping # measuring but we aren't dumping
          # Estimate whether we'll make it to another measurement before hitting WTL. If not, flush and restart.
          to_udsweep = mc.a.to["udsweep"]
          udsd = TimerOutputs.time(to_udsweep) *10.0^(-9)/TimerOutputs.ncalls(to_udsweep)
          secs_to_meas = p.write_every_nth * udsd
          secs_to_meas *= 1.1 # add 10 percent because we might be slower
          next_meas_date = now() + Millisecond(ceil(Int, secs_to_meas*1000))

          if next_meas_date >= p.walltimelimit
            println("Approaching wall-time limit. Won't make it to next measurement. Safely exiting. (i = $(i)). Current date: $(Dates.format(now(), "d.u yyyy HH:MM")).")
            println("Flushing configurations which haven't been dumped yet.")
            flush(configurations)
            flush(greens)
            flush(occ)
            # flush(boson_action)
            saverng(p.output_file; group="resume/rng")
            exit(42)
          end
        end
      end
    end

    if mod(i, 100) == 0
      a.acc_rate = a.acc_rate / (100 * 2 * p.slices)
      a.acc_rate_global = a.acc_rate_global / (100 / p.global_rate)
      println("\t", i*2)
      @printf("\t\tsweep dur (total mean): %.4fs\n", TimerOutputs.time(a.to["udsweep"])/2 *10.0^(-9)/TimerOutputs.ncalls(a.to["udsweep"]))
      @printf("\t\tacc rate (local) : %.1f%%\n", a.acc_rate*100)
      if p.global_updates
        @printf("\t\tacc rate (global): %.1f%%\n", a.acc_rate_global*100)
        @printf("\t\tacc rate (global, overall): %.1f%%\n", a.acc_global/a.prop_global*100)
      end
      a.acc_rate = 0.0
      a.acc_rate_global = 0.0
      flush(stdout)
    end

    if (approaching_wtl = now() >= p.walltimelimit)
      println("Approaching wall-time limit. Safely exiting immediately. (i = $(i)). Current date: $(Dates.format(now(), "d.u yyyy HH:MM")).")
      println("Won't flush because state wouldn't be resumable.")
      exit(42)
    end
  end

  println("Final flush.")
  flush(configurations)
  # flush(boson_action)
  flush(greens)
  flush(occ)

  nothing
end

function update(mc::DQMC, i::Int)
  p = mc.p
  s = mc.s
  l = mc.l
  a = mc.a

  propagate(mc)

  if p.global_updates && (s.current_slice == p.slices && s.direction == -1 && mod(i, p.global_rate) == 0)
    a.prop_global += 1
    @mytimeit a.to "global updates" b = global_update(mc)
    a.acc_rate_global += b
    a.acc_global += b
  end

  a.acc_rate += local_updates(mc)
  nothing
end



# -------------------------------------------------------
#                     Other stuff
# -------------------------------------------------------
# cosmetics
import Base.summary
import Base.show
Base.summary(mc::DQMC) = "DQMC"
function Base.show(io::IO, mc::DQMC{C}) where C<:Checkerboard
  print(io, "DQMC of O($(mc.p.opdim)) model\n")
  print(io, "r = ", mc.p.r, ", λ = ", mc.p.lambda, ", c = ", mc.p.c, ", u = ", mc.p.u, "\n")
  print(io, "Beta: ", mc.p.beta, " (T ≈ $(round(1/mc.p.beta, digits=3)))", "\n")
  print(io, "Checkerboard: ", C, "\n")
  print(io, "B-field: ", mc.p.Bfield)
end
Base.show(io::IO, m::MIME"text/plain", mc::DQMC) = print(io, mc)



"""
Calculate DateTime where wall-time limit will be reached.

Example call: wtl2DateTime("3-12:42:05", now())
"""
function wtl2DateTime(wts::AbstractString, start_time::DateTime)
  @assert occursin("-", wts)
  @assert occursin(":", wts)
  @assert length(wts) >= 10

  tmp = split(wts, "-")

  d = parse(Int, tmp[1])
  h, m, s = parse.(Int, split(tmp[2], ":"))

  start_time + Dates.Day(d) + Dates.Hour(h) + Dates.Minute(m) + Dates.Second(s)
end

function set_walltimelimit!(p, start_time)
  if "WALLTIMELIMIT" in keys(ENV)
    p.walltimelimit = wtl2DateTime(ENV["WALLTIMELIMIT"], start_time)
    @show ENV["WALLTIMELIMIT"]
    @show p.walltimelimit
  elseif occursin("cheops", gethostname())
    p.walltimelimit = wtl2DateTime("9-23:30:00", start_time) # CHEOPS
    println("Set CHEOPS walltime limit, i.e. 9-23:30:00.")
  elseif occursin("jw", gethostname())
    p.walltimelimit = wtl2DateTime("0-23:30:00", start_time) # JUWELS
    println("Set JUWELS walltime limit, i.e. 0-23:30:00.")
  end

  nothing
end


"""

    csheuristics(wctl, udsd, writeeverynth; maxcs=100)

Find a reasonable value for the chunk size `cs` based on wallclocktime limit (DateTime),
ud-sweep duration `udsd` (in seconds) and `writeeverynth` to avoid

* non-progressing loop of resubmitting the simulation because we never make it to a dump
* memory overflow (we use a maximum chunk size `maxcs`)
* slowdown of simulation by IO (for too small chunk size)

A value of 0 for `wctl` is interpreted as no wallclocktime limit and chunk size `maxcs` will be returned.
"""
function csheuristics(wctl::DateTime, udsd::Real, writeeverynth::Int; maxcs::Int=100)
    wcts = (wctl - now()).value / 1000. # seconds till we hit the wctl
    # wcts == 0.0 && (return maxcs) # no wallclocktime limit

    num_uds = wcts/(udsd * writeeverynth) # Float64: number of add!s we will (in theory) perform in the given time
    num_uds *= 0.80 # 20% buffer, i.e. things might take longer than expected.
    cs = max(floor(Int, min(num_uds, 100)), 1)
end


"""
Choose a apropriate chunk size for the given Monte Carlo simulation.
"""
function choose_chunk_size(mc::AbstractDQMC)
    p = mc.p
    if !("udsweep" in keys(mc.a.to.inner_timers))
      @warn "No \"udsweep\" timer found. Choosing default `cs=100`!! Job might never dump to disk!"
      return 100
    end
    to_udsweep = mc.a.to["udsweep"]

    udsd = TimerOutputs.time(to_udsweep) *10.0^(-9)/TimerOutputs.ncalls(to_udsweep)
    cs = csheuristics(p.walltimelimit, udsd, p.write_every_nth)
    cs = min(cs, floor(Int, p.measurements/p.write_every_nth)) # cs musn't be larger than # measurements
    p.edrun && (cs = 1000) # this should probably better set maxcs in csheuristics call

    secs_to_dump = cs * p.write_every_nth * udsd
    dump_date = now() + Millisecond(ceil(Int, secs_to_dump*1000))
    println("Chose a chunk size of $cs. Should dump to file around $(formatdate(dump_date)). Walltime limit is $(formatdate(p.walltimelimit)).")

    return cs
end


formatdate(d) = Dates.format(d, "d.u yyyy HH:MM")


"""
Draw a random number from a uniform distribution over an interval [-b, b].
"""
@inline randuniform(b::Float64) = -b + 2 * b * rand() # taken from Distributions.jl: https://tinyurl.com/ycr9jnt4
randuniform(b::Float64, d::Int) = begin
    x = Vector{Float64}(undef, d)
    @inbounds for k in Base.OneTo(d)
      x[k] = randuniform(b)
    end
    x
end



"""
Manual and incomplete(!) serializer for JLD.

It only works for a single most inner timer for which `isempty(to.inner_timers) == true`.
"""
struct MyTimerOutputSerializer
    td::TimerOutputs.TimeData
    name::String
end

JLD.writeas(to::TimerOutput) = MyTimerOutputSerializer(to.accumulated_data, to.name)

function JLD.readas(tos::MyTimerOutputSerializer)
    to = TimerOutput()
    @timeit to tos.name 3+3 # something, will overwrite it
    to[tos.name].accumulated_data = tos.td
    to
end