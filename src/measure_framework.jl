using Pkg
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



# #######################################################
#                       Framework
# #######################################################
# -------------------------------------------------------
#                       Includes
# -------------------------------------------------------
global const TIMING = true
include("dqmc_framework.jl")
using Parameters
using ProgressMeter
using TimerOutputs
using BinningAnalysis


const OBSERVABLES = Set((:chi_dyn,
                         :chi_dyn_symm,
                         :binder,
                         :etpc,
                         :zfpc,
                         :etcdc,
                         :zfcdc,
                         :sfdensity
))

const PLUSMINUS_OBSERVABLES = Set((:etpc,
                                   :zfpc,
                                   :etcdc,
                                   :zfcdc
))


const CALC_IN_ONE_GO = Set((:chi_dyn,       # always in one go
                            :chi_dyn_symm,  # always in one go
                            :binder         # always in one go
))





# -------------------------------------------------------
#                  Define Macros
# -------------------------------------------------------
macro addlightobs(name, zero)
  nc = esc(:(num_confs))
  o_inonego = esc(:(obs_inonego))
  o_insteps = esc(:(obs_insteps))
  mp = esc(:(mp))
  n = "$name"
  z = esc(:($zero))
  return quote
    if Symbol($n) in $(mp).todo_insteps
      $(o_insteps) = add($(o_insteps), $name = create_or_load_lightobs($n, $z, name = $n, alloc = $nc))
    else
      $(o_inonego) = add($(o_inonego), $name = LightObservable($z, name = $n, alloc = $nc))
    end
  end
end

macro addobs(name, T)
  nc = esc(:(num_confs))
  o_inonego = esc(:(obs_inonego))
  o_insteps = esc(:(obs_insteps))
  mp = esc(:(mp))
  n = "$name"
  t = esc(:($T))
  return quote 
    if Symbol($n) in $(mp).todo_insteps
      $(o_insteps) = add($(o_insteps), $name = create_or_load_obs($n, $t, name = $n, alloc = $nc))
    else
      $(o_inonego) = add($(o_inonego), $name = Observable($t, name = $n, alloc = $nc))
    end
  end
end





# -------------------------------------------------------
#                  Define MeasParams
# -------------------------------------------------------
@with_kw mutable struct MeasParams
  requested::Set{Symbol} = Set{Symbol}()
  todo::Set{Symbol} = Set{Symbol}()
  todo_inonego::Set{Symbol} = Set{Symbol}()
  todo_insteps::Set{Symbol} = Set{Symbol}()

  # input & output files
  meas_infile::String = ""
  outfile::String = ""
  p::Params = Params() # DQMC Params
  dqmc_outfile::String = ""

  # parameters
  overwrite::Bool = false
  num_threads::Int = 1
  walltimelimit::Dates.DateTime = Dates.DateTime("2099", "YYYY")
  save_after::Int = 10
  to::TimerOutput = TimerOutput()
  confs_iterator_start::Int = 1 # this is the S in "for i in S:length(confs)"
end



function measxml2MeasParams(fname)
  # meas.xml: direct mapping of xml fields to kwargs
  mpdict = Dict(Symbol(lowercase(k))=>lowercase(v) for (k,v) in xml2dict(fname, false))
  mp = MeasParams()

  for (k, v) in mpdict
    if k in OBSERVABLES
      if parse(Bool, v)
        if k in PLUSMINUS_OBSERVABLES
          push!(mp.requested, Symbol(string(k)*"_minus"))
          push!(mp.requested, Symbol(string(k)*"_plus"))
        else
          push!(mp.requested, k)
        end
      end

    elseif k == :overwrite
      mp.overwrite = parse(Bool, v)
    elseif k == :num_threads
      mp.num_threads = parse(Int64, v)
    elseif k == :save_after
      mp.save_after = parse(Int64, v)
    end
  end

  mp.meas_infile = fname

  mp.dqmc_outfile = replace(mp.meas_infile, ".meas.xml" => ".out.h5")
  if !isfile(mp.dqmc_outfile)
      mp.dqmc_outfile = replace(mp.dqmc_outfile, ".out.h5" => ".out.h5.running")
      isfile(mp.dqmc_outfile) || error("Couldn't find .out.h5[.running] file.")
  end
  mp.outfile = replace(mp.dqmc_outfile, ".out.h5" => ".meas.h5")

  dqmc_infile = fname[1:end-9]*".in.xml"
  if isfile(dqmc_infile)
    println("Loading DQMC parameters (.in.xml)")
    xml2parameters!(mp.p, dqmc_infile)
  else
    println("Loading DQMC parameters (.out.h5)")
    try
      hdf52parameters!(mp.p, mp.dqmc_outfile)
    catch er
      println("Error while loading parameters from .out.h5 file")
      throw(er)
    end
  end

  if @isdefined start_time
    set_walltimelimit!(mp, start_time)
  else
    set_walltimelimit!(mp, now())
  end

  return mp
end








@inline function need_to_setup_mc(mp::MeasParams)
  symb = (:etpc_plus, :zfpc_plus, :etcdc_plus, :zfcdc_plus, :sfdensity)
  return any(in.(symb, Ref(mp.todo)))
end
@inline function need_to_load_etgf(mp::MeasParams)
  symb = (:etpc_plus, :etcdc_plus, :zfcdc_plus, :sfdensity)
  return any(in.(symb, Ref(mp.todo)))
end
@inline function need_to_meas_tdgfs(mp::MeasParams)
  symb = (:zfpc_plus, :zfcdc_plus, :sfdensity)
  return any(in.(symb, Ref(mp.todo)))
end
@inline function need_to_meas_zfccc(mp::MeasParams)
  symb = (:sfdensity)
  return any(in.(symb, Ref(mp.todo)))
end





@inline function need_to_load_etgf(ol::NamedTuple{K,V}) where {K,V}
  symb = (:etpc_plus, :etcdc_plus, :zfcdc_plus, :sfdensity)
  return any(Base.sym_in.(symb, Ref(K)))
end
@inline function need_to_meas_tdgfs(ol::NamedTuple{K,V}) where {K,V}
  symb = (:zfpc_plus, :zfcdc_plus, :sfdensity)
  return any(Base.sym_in.(symb, Ref(K)))
end
@inline function need_to_meas_zfccc(ol::NamedTuple{K,V}) where {K,V}
  return Base.sym_in(:sfdensity, K)
end








# -------------------------------------------------------
#              Initialize stack for meas
# -------------------------------------------------------
function initialize_stack_for_measurements(mc::AbstractDQMC, mp::MeasParams)
    _initialize_stack(mc)

    # allocate for measurement run based on todos
    :etpc_plus in mp.todo  && allocate_etpc!(mc)
    :zfpc_plus in mp.todo  && allocate_zfpc!(mc)
    :etcdc_plus in mp.todo && allocate_etcdc!(mc)
    :zfcdc_plus in mp.todo && allocate_zfcdc!(mc)

    need_to_meas_zfccc(mp) && allocate_zfccc!(mc)
    need_to_meas_tdgfs(mp) && allocate_tdgfs!(mc)

    nothing
end












"""
Groups todos according to whether they should be calculated
 * in one go
 * step by step with intermediate saves
"""
function create_todolist!(mp::MeasParams)
  mp.todo = Set{Symbol}()

  if !endswith(mp.outfile, ".running") && isfile(mp.outfile*".running")
    # job has finished since last measurement run
    mv(mp.outfile*".running", mp.outfile)
  end

  if isfile(mp.outfile) && !mp.overwrite
    # Start/continue unfinished measurements but don't overwrite anything
    println("Measurement file found.");
    println("Checking what is left todo...")

    local confs
    try
      confs = loadobs_frommemory(mp.dqmc_outfile, "obs/configurations")
    catch err
      println("Couldn't read configuration data. Probably no configurations yet? Exiting.")
      exit()
    end
    allobs = listobs(mp.outfile)
    N = length(confs)
    counts = Dict{Symbol, Int64}()

    jldopen(mp.outfile, "r") do f
      for o in mp.requested

        o == :binder && continue # special case binder cumulant

        # IN-ONE-GO
        if strip_plusminus(o) in CALC_IN_ONE_GO
          p = joinpath("obs/", string(o))

          if !HDF5.has(f.plain, p)
            push!(mp.todo, o)
          else
            count = read(f[joinpath(p, "count")])
            if count != N
              push!(mp.todo, o)
              # counts[o] = count
            end
          end
        else
          # IN-STEPS
          p = joinpath("obj/", string(o))

          if !HDF5.has(f.plain, p)
            push!(mp.todo, o)
          else
            count = length(read(f[p]))
            if count != N
              push!(mp.todo, o)
              counts[o] = count
            end
          end
        end

      end # observable loop
    end # open file

    if length(unique(values(counts))) > 1
      display(counts)
      flush(stdout)
      @error "Found unfinished measurements of different length (not yet supported). Aborting."
      # TODO: instead of abort, start from scratch, i.e. overwrite finite length unfinished measurements.
    end

    # Special case: binder
    if :binder in mp.requested && :chi_dyn in mp.todo
      push!(mp.todo, :binder)
    end

    mp.confs_iterator_start = try unique(values(counts))[1] + 1 catch er 1 end

  else # there is no meas outfile yet or we are in overwrite mode
    mp.todo = copy(mp.requested)
  end

  mp.todo
end


function group_todos!(mp)
  mp.todo_inonego

  for o in mp.todo
    if strip_plusminus(o) in CALC_IN_ONE_GO
      push!(mp.todo_inonego, o)
    else
      push!(mp.todo_insteps, o)
    end
  end

  nothing
end

strip_plusminus(str::AbstractString) = replace(replace(str, "_plus" => ""), "_minus" => "")
strip_plusminus(s::Symbol) = Symbol(strip_plusminus(string(s)))



# -------------------------------------------------------
#                        Main
# -------------------------------------------------------
function measure(mp::MeasParams; debug=false)

  create_todolist!(mp)

  if isempty(mp.todo)
    println("Nothing on the todo list. Exiting.")
    exit()
  end

  group_todos!(mp)

  @show mp.todo_inonego
  @show mp.todo_insteps
  flush(stdout)

  # Overwrite mode?
  if isfile(mp.outfile) && mp.overwrite
    println("\nOVERWRITE MODE")
    jldopen(mp.outfile, "r+") do f
      for o in mp.todo
        p = joinpath("obj/", string(o))
        if HDF5.has(f.plain, p)
          println("Clearing $p")
          o_delete(f, p)
        end
        p = joinpath("obs/", string(o))
        if HDF5.has(f.plain, p)
          println("Clearing $p")
          o_delete(f, p)
        end
      end
    end
    println()
  end


  # -------------- Load DQMC params/results -----------------
  println("\nLoading DQMC results...")

  greens = nothing; 
  mc = nothing;
  local confs

  confs = loadobs_frommemory(mp.dqmc_outfile, "obs/configurations")

  if need_to_load_etgf(mp)
    if "greens" in listobs(mp.dqmc_outfile)
      greens = loadobs_frommemory(mp.dqmc_outfile, "obs/greens")
    else
      error("Couldn't find ETGF in $(mp.dqmc_outfile) although need_to_load_etgf == true.")
      # TODO: Calculate ETGF if it isn't found in dqmc output.
    end
  end

  # prepare dqmc framework if necessary
  if need_to_setup_mc(mp)
    mc = DQMC(mp.p)
    initialize_stack_for_measurements(mc, mp)
  end


  # ------------------- Create list of observables to measure --------------------
  println("\nPreparing observables......................")
  num_confs = length(confs)
  nsweeps = num_confs * mp.p.write_every_nth
  obs_inonego = NamedTuple()
  obs_insteps = NamedTuple()

  # chi_dyn
  if :chi_dyn_symm in mp.todo
    @addobs chi_dyn_symm Array{Float64, 3}
  end

  if :chi_dyn in mp.todo
    @addobs chi_dyn Array{Float64, 3}
  end
  
  # binder
  if :binder in mp.todo
    @addobs m2s Float64
    @addobs m4s Float64
    obs_inonego = add(obs_inonego, binder = Observable(Float64, name="binder"))
  end

  # etpc
  if :etpc_plus in mp.todo
    zero_etpc = zeros(Float64, mp.p.L, mp.p.L)
    @addlightobs etpc_plus zero_etpc
    @addlightobs etpc_minus zero_etpc
  end

  # zfpc
  if :zfpc_plus in mp.todo
    zero_zfpc = zeros(Float64, mp.p.L, mp.p.L)
    @addlightobs zfpc_plus zero_zfpc
    @addlightobs zfpc_minus zero_zfpc
  end

  # etcdc
  if :etcdc_plus in mp.todo
    zero_etcdc = zeros(ComplexF64, mp.p.L, mp.p.L)
    @addlightobs etcdc_plus zero_etcdc
    @addlightobs etcdc_minus zero_etcdc
  end

  # zfcdc
  if :zfcdc_plus in mp.todo
    zero_zfcdc = zeros(ComplexF64, mp.p.L, mp.p.L)
    @addlightobs zfcdc_plus zero_zfcdc
    @addlightobs zfcdc_minus zero_zfcdc
  end

  # sfdensity
  if :sfdensity in mp.todo
    @addlightobs sfdensity Float64
  end

  # # tdgfs
  # if mp.tdgfs
  #   # create zero element for LightObservable
  #   Nflv = mc.p.flv * mc.l.sites
  #   tdgf_size = (Nflv, Nflv, mc.p.slices)
  #   zero_tdgf = zeros(geltype(mc), tdgf_size)
  #   (obs = add(obs, tdgfs_Gt0 = LightObservable(zero_tdgf, name="Time-displaced Green's function G(tau,0) (TDGF Gt0)", alloc=num_confs)))
  #   (obs = add(obs, tdgfs_G0t = LightObservable(zero_tdgf, name="Time-displaced Green's function G(0, tau) (TDGF G0t)", alloc=num_confs)))
  # end



  print("........................... Done. ");
  println("\n")
  @show keys(obs_inonego)
  @show keys(obs_insteps)
  println("\n")



  # --------------------- Measure -------------------------
  if length(obs_inonego) > 0 && !debug
    reset_timer!(mp.to)
    measure_inonego(mp, obs_inonego, confs, greens, mc)
    export_results(mp, obs_inonego, nsweeps)
  end

  if length(obs_insteps) > 0 && !debug
    reset_timer!(mp.to)
    measure_insteps(mp, obs_insteps, confs, greens, mc)
    export_results(mp, obs_insteps, nsweeps)
  end

  debug && (return obs_inonego, obs_insteps)

  nothing
end









# -------------------------------------------------------
#                 MEASUREMENT LOOPS
# -------------------------------------------------------
function measure_inonego(mp::MeasParams, obs::NamedTuple{K,V}, confs, greens, mc) where {K,V}
    length(obs) > 0 && println("---------------------- Measuring (in-one-go) ----------------------");
    flush(stdout)

    @mytimeit mp.to "measure loop" begin
      @inbounds @views @showprogress for i in 1:length(confs)
          # println(i); flush(stdout);
          @mytimeit mp.to "load conf" conf = confs[i]

          @mytimeit mp.to "bosonic" measure_bosonic(mp, obs, conf, i)

          @mytimeit mp.to "load etgf" g = need_to_load_etgf(obs) ? greens[i] : nothing # load single greens from disk per MCO.jl
          @mytimeit mp.to "fermionic" measure_fermionic(mp, obs, conf, g, mc, i)
      end
    end

    if TIMING
      println()
      display(mp.to)
      println()
      flush(stdout)
    end

    if :binder in keys(obs)
        # binder postprocessing
        m2ev2 = mean(obs[:m2s])^2
        m4ev = mean(obs[:m4s])

        push!(obs[:binder], m4ev/m2ev2)
    end
end



function measure_insteps(mp::MeasParams, obs::NamedTuple{K,V}, confs, greens, mc) where {K,V}
    length(obs) > 0 && println("---------------------- Measuring (in-steps) ----------------------");
    flush(stdout)

    @inbounds @views @showprogress for i in mp.confs_iterator_start:length(confs)
        @mytimeit mp.to "iteration" begin
          println(i); flush(stdout);
          # sleep(2)
          @mytimeit mp.to "load conf" conf = confs[i]

          @mytimeit mp.to "bosonic" measure_bosonic(mp, obs, conf, i)

          @mytimeit mp.to "load etgf" g = need_to_load_etgf(obs) ? greens[i] : nothing # load single greens from disk per MCO.jl
          @mytimeit mp.to "fermionic" measure_fermionic(mp, obs, conf, g, mc, i)


          @mytimeit mp.to "intermediate save" if mod1(i, mp.save_after) == mp.save_after
            save_obs_objects(mp, obs)

            # Estimate whether we'll make it to another save before hitting WTL. If not, flush and restart.
            titer = mp.to["iteration"]
            iterdur = TimerOutputs.time(titer) *10.0^(-9)/TimerOutputs.ncalls(titer)
            secs_to_save = mp.save_after * iterdur
            secs_to_save *= 1.1 # add 10 percent buffer
            next_save_date = now() + Millisecond(ceil(Int, secs_to_save * 1000))

            if next_save_date >= mp.walltimelimit
              println("Approaching wall-time limit. Won't make it to next save.")
              println("Exporting intermediate results.")
              export_results(mp, obs, nothing)
              println("Safely exiting. (i = $(i)). Current date: $(Dates.format(now(), "d.u yyyy HH:MM")).")
              exit(42)
            end
          end # if save
        end # time one iteration
    end

    @mytimeit mp.to "final save" save_obs_objects(mp, obs)

    if TIMING
      println()
      display(mp.to)
      println()
      flush(stdout)
    end
end

function save_obs_objects(mp, obs)
  println("Intermediate save..."); flush(stdout)
  jldopen(mp.outfile, isfile(mp.outfile) ? "r+" : "w") do f
    for (o, obj) in pairs(obs)
      p = joinpath("obj/", string(o))
      HDF5.has(f.plain, p) && JLD.o_delete(f, p)
      f[p] = obj
    end
  end
  # OPT: Be smarter for Observable objects (bosonic observables)
  nothing
end







# -------------------------------------------------------
#                      MEASUREMENTS
# -------------------------------------------------------
function measure_bosonic(mp, obs, conf, i)
      # chi_dyn
      @mytimeit mp.to "chi" if :chi_dyn in keys(obs)
        chi = measure_chi_dynamic(conf)
        push!(obs[:chi_dyn], chi)

        if :chi_dyn_symm in keys(obs)
          chi = (permutedims(chi, [2,1,3]) + chi)/2 # C4 is basically flipping qx and qy (which only go from 0 to pi since we perform a real fft.)
          push!(obs[:chi_dyn_symm], chi)
        end
      end

      # binder
      @mytimeit mp.to "binder" if :binder in keys(obs)
        m = mean(conf, dims=(2,3))
        push!(obs[:m2s], dot(m, m))
        push!(obs[:m4s], last(obs[:m2s])^2)
      end
      nothing
end





function measure_fermionic(mp, obs::NamedTuple{K,V}, conf, greens, mc, i) where {K,V}
    if !isnothing(mc)
      mc.p.hsfield = conf
    end

    # etpc
    @mytimeit mp.to "etpc" if :etpc_plus in K
        measure_etpc!(mc, greens)
        push!(obs[:etpc_plus], mc.s.meas.etpc_plus)
        push!(obs[:etpc_minus], mc.s.meas.etpc_minus)
    end

    # etcdc
    @mytimeit mp.to "etcdc" if :etcdc_plus in K
        measure_etcdc!(mc, greens)
        push!(obs[:etcdc_plus], mc.s.meas.etcdc_plus)
        push!(obs[:etcdc_minus], mc.s.meas.etcdc_minus)
    end

    @mytimeit mp.to "calculate TDGFs" if need_to_meas_tdgfs(obs)
      measure_tdgfs!(mc)
      GC.gc()
      Gt0 = mc.s.meas.Gt0
      G0t = mc.s.meas.G0t
    end

    @mytimeit mp.to "zfccc" if need_to_meas_zfccc(obs)
      measure_zfccc!(mc, greens, Gt0, G0t)
      zfccc = mc.s.meas.zfccc
    end

    # zfpc
    @mytimeit mp.to "zfpc" if :zfpc_plus in K
        measure_zfpc!(mc, Gt0)
        push!(obs[:zfpc_plus], mc.s.meas.zfpc_plus)
        push!(obs[:zfpc_minus], mc.s.meas.zfpc_minus)
    end

    # zfcdc
    @mytimeit mp.to "zfcdc" if :zfcdc_plus in K
        measure_zfcdc!(mc, greens, Gt0, G0t)
        push!(obs[:zfcdc_plus], mc.s.meas.zfcdc_plus)
        push!(obs[:zfcdc_minus], mc.s.meas.zfcdc_minus)
    end

    @mytimeit mp.to "sfdensity" if :sfdensity in K
        push!(obs[:sfdensity], measure_sfdensity(mc, zfccc))
    end

    # # tdgfs
    # if :tdgfs_Gt0 in keys(obs)        
    #     push!(obs[:tdgfs_Gt0], VectorOfArray(Gt0))
    #     push!(obs[:tdgfs_G0t], VectorOfArray(G0t))
    # end

    nothing
end








# -------------------------------------------------------
#                      Export
# -------------------------------------------------------
function export_results(mp, obs, nsweeps)
    println("Calculating errors and exporting..."); flush(stdout)

    # bosonic
    :chi_dyn in keys(obs) && export_result(obs[:chi_dyn], mp.outfile, "obs/chi_dyn"; timeseries=true)
    :chi_dyn_symm in keys(obs) && export_result(obs[:chi_dyn_symm], mp.outfile, "obs/chi_dyn_symm"; timeseries=true)
    :binder in keys(obs) && export_result(obs[:binder], mp.outfile, "obs/binder", error=false) # jackknife for error

    # fermionic
    :etpc_plus in keys(obs) && export_result(obs[:etpc_plus], mp.outfile, "obs/etpc_plus")
    :etpc_minus in keys(obs) && export_result(obs[:etpc_minus], mp.outfile, "obs/etpc_minus")
    :zfpc_plus in keys(obs) && export_result(obs[:zfpc_plus], mp.outfile, "obs/zfpc_plus")
    :zfpc_minus in keys(obs) && export_result(obs[:zfpc_minus], mp.outfile, "obs/zfpc_minus")
    :etcdc_plus in keys(obs) && export_result(obs[:etcdc_plus], mp.outfile, "obs/etcdc_plus")
    :etcdc_minus in keys(obs) && export_result(obs[:etcdc_minus], mp.outfile, "obs/etcdc_minus")
    :zfcdc_plus in keys(obs) && export_result(obs[:zfcdc_plus], mp.outfile, "obs/zfcdc_plus")
    :zfcdc_minus in keys(obs) && export_result(obs[:zfcdc_minus], mp.outfile, "obs/zfcdc_minus")
    :sfdensity in keys(obs) && export_result(obs[:sfdensity], mp.outfile, "obs/sfdensity")
    # :tdgfs_Gt0 in keys(obs) && export_result(obs[:tdgfs_Gt0], mp.outfile, "obs/tdgfs_Gt0")
    # :tdgfs_G0t in keys(obs) && export_result(obs[:tdgfs_G0t], mp.outfile, "obs/tdgfs_G0t")

    # meta data
    if !isnothing(nsweeps)
      h5open(mp.outfile, "r+") do fout
          HDF5.has(fout, "nsweeps") && HDF5.o_delete(fout, "nsweeps")
          HDF5.has(fout, "write_every_nth") && HDF5.o_delete(fout, "write_every_nth")
          fout["nsweeps"] = nsweeps
          fout["write_every_nth"] = mp.p.write_every_nth
      end
    end
    println("Done. Exported to $(mp.outfile).")
    flush(stdout)
end












# -------------------------------------------------------
#                 Helpers and stuff
# -------------------------------------------------------

"""
Allows one to use keyword syntax to add new entries to a `NamedTuple` `obs`.
"""
add(obs::NamedTuple; kw...) = combine(obs, values(kw))
"""
Combine two `NamedTuple`s into a new one.
"""
combine(x::NamedTuple, y::NamedTuple) = NamedTuple{(keys(x)..., keys(y)...)}((x...,y...))


"""
Create or load LightObservable
"""
function create_or_load_lightobs(abbrev::String, args...; kwargs...)
  print(abbrev, ": looking for old LightObservable ... ")
  if isfile(mp.outfile)
    x = nothing
    jldopen(mp.outfile, "r") do f
      p = joinpath("obj/", abbrev)
      if HDF5.has(f.plain, p)
        x = read(f[p])
        println("found and loaded!")
      end
    end
    !isnothing(x) && return x
  end

  # couldn't be loaded -> create new LightObservable
  # println("created new lightobs")
  println("not found.")
  return LightObservable(args...; kwargs...)
end

"""
Create or load Observable
"""
function create_or_load_obs(abbrev::String, args...; kwargs...)
  print(abbrev, ": looking for old Observable ... ")
  if isfile(mp.outfile)
    x = nothing
    jldopen(mp.outfile, "r") do f
      p = joinpath("obj/", abbrev)
      if HDF5.has(f.plain, p)
        x = read(f[p])
        println("found and loaded!")
      end
    end
    !isnothing(x) && return x
  end

  # couldn't be loaded -> create new LightObservable
  println("not found.")
  return Observable(args...; kwargs...)
end

nothing