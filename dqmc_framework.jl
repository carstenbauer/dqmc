# -------------------------------------------------------
#                    Abstract types
# -------------------------------------------------------
abstract type Checkerboard end
abstract type CBTrue <: Checkerboard end
abstract type CBFalse <: Checkerboard end
abstract type CBGeneric <: CBTrue end
abstract type CBAssaad <: CBTrue end

abstract type AbstractDQMC{C<:Checkerboard, GreensEltype<:Number, HoppingEltype<:Number} end

isdefined(:DQMC_CBTrue) || (global const DQMC_CBTrue = AbstractDQMC{C} where C<:CBTrue)
isdefined(:DQMC_CBFalse) || (global const DQMC_CBFalse = AbstractDQMC{C} where C<:CBFalse)



# -------------------------------------------------------
#                    Includes
# -------------------------------------------------------
using Helpers
using MonteCarloObservable
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



# -------------------------------------------------------
#                    Concrete types
# -------------------------------------------------------
mutable struct Analysis
  acc_rate::Float64
  acc_rate_global::Float64
  prop_global::Int
  acc_global::Int

  Analysis() = new()
end

mutable struct DQMC{C<:Checkerboard, GreensEltype<:Number, HoppingEltype<:Number} <: AbstractDQMC{C, GreensEltype, HoppingEltype}
  p::Params
  l::Lattice{HoppingEltype}
  s::Stack{GreensEltype}
  a::Analysis
end

DQMC(p::Params) = begin
  CB = CBFalse
  p.chkr && (CB = iseven(p.L) ? CBAssaad : CBGeneric)

  ### SET DATATYPES
  G = Complex128
  H = Complex128
  if !p.Bfield
    H = Float64;
    G = p.opdim > 1 ? Complex128 : Float64; # O(1) -> real GF
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
  srand(mc.p.seed); # init RNG

  # Init hsfield
  println("\nInitializing HS field")
  mc.p.hsfield = rand(mc.p.opdim,mc.l.sites,mc.p.slices)
  println("Initializing boson action\n")
  mc.p.boson_action = calculate_boson_action(mc)

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

# TODO resumeshort works, resumebroken not. Why?
function resume!(mc::DQMC, lastconf, prevmeasurements::Int)
  const p = mc.p

  # Init hsfield
  println("\nLoading last HS field")
  p.hsfield = deepcopy(lastconf)
  println("Initializing boson action\n")
  p.boson_action = calculate_boson_action(mc)

  h5open(p.output_file, "r") do f
    box = read(f, "resume/box")
    box_global = read(f, "resume/box_global")
    p.box = Uniform(-box, box)
    p.box_global = Uniform(-box_global, box_global)
  end

  println("\n\nMC Measure (resuming) - ", p.measurements, " (total $(p.measurements + prevmeasurements))")
  flush(STDOUT)
  measure!(mc, prevmeasurements)

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

function measure!(mc::DQMC, prevmeasurements=0)
  const a = mc.a
  const l = mc.l
  const p = mc.p
  const s = mc.s

  initialize_stack(mc)
  println("Renewing stack")
  build_stack(mc)
  println("Initial propagate: ", s.current_slice, " ", s.direction)
  propagate(mc)

  cs = min(floor(Int, p.measurements/p.write_every_nth), 100)

  configurations = Observable(typeof(p.hsfield), "configurations"; alloc=cs, inmemory=false, outfile=p.output_file, dataset="obs/configurations")
  greens = Observable(typeof(mc.s.greens), "greens"; alloc=cs, inmemory=false, outfile=p.output_file, dataset="obs/greens")
  # fermion density
  chi_inv_dynamic = Observable(Array{Float64,3}, "chi inverse dynamic (qy, qx, iomega)"; alloc=cs, inmemory=false, outfile=p.output_file, dataset="obs/chi_inv_dynamic")
  chi = Observable(Float64, "chi"; alloc=cs, inmemory=false, outfile=p.output_file, dataset="obs/chi")
  chi_inv = Observable(Float64, "chi inverse"; alloc=cs, inmemory=false, outfile=p.output_file, dataset="obs/chi_inv")
  boson_action = Observable(Float64, "boson_action"; alloc=cs, inmemory=false, outfile=p.output_file, dataset="obs/boson_action")

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
        dumping = (length(boson_action)+1)%cs == 0
        dumping && println("Dumping...")
        # @time begin
        add!(boson_action, p.boson_action)

        chi_dyn = measure_chi_dynamic(mc.p.hsfield)
        chi_inv_dyn = 1./chi_dyn
        add!(chi_inv_dynamic, chi_inv_dyn)
        add!(chi_inv, chi_inv_dyn[1,1,1])
        add!(chi, chi_dyn[1,1,1])

        add!(configurations, p.hsfield)
        add!(greens, effective_greens2greens(mc, s.greens))

        saverng(p.output_file; group="resume/rng")
        dumping && println("Dumping block of $cs datapoints was a success")
        flush(STDOUT)
        # end
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

  # finish measurements, i.e. calculate errors
  MonteCarloObservable.export_error(greens)
  MonteCarloObservable.export_error(chi_inv_dynamic)
  MonteCarloObservable.export_error(chi_inv)
  MonteCarloObservable.export_error(chi)
  MonteCarloObservable.export_error(boson_action)

  # export_result(boson_action, p.output_file, "obs/boson_action")
  # export_result(chi, p.output_file, "obs/chi")
  # export_result(chi_inv, p.output_file, "obs/chi_inv")

  toq();
  nothing
end

function update(mc::DQMC, i::Int)
  const p = mc.p
  const s = mc.s
  const l = mc.l
  const a = mc.a

  propagate(mc)

  if p.global_updates && (s.current_slice == p.slices && s.direction == -1 && mod(i, p.global_rate) == 0)
    a.prop_global += 1
    b = global_update(mc)
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
  print(io, "Beta: ", mc.p.beta, " (T ≈ $(round(1/mc.p.beta, 3)))", "\n")
  print(io, "Checkerboard: ", C, "\n")
  print(io, "B-field: ", mc.p.Bfield)
end
Base.show(io::IO, m::MIME"text/plain", mc::DQMC) = print(io, mc)