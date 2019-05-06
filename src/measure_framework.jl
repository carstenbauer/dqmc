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


const OBSERVABLES = (:chi_dyn,
                     :chi_dyn_symm,
                     :binder,
                     :etpc,
                     :zfpc,
                     :etcdc,
                     :zfcdc,
                     :sfdensity
)

const PLUSMINUS_OBSERVABLES = (:etpc,
                               :zfpc,
                               :etcdc,
                               :zfcdc
)





# -------------------------------------------------------
#                  Define Macros
# -------------------------------------------------------
macro addlightobs(name, zero)
  nc = esc(:(num_confs))
  o = esc(:(obs))
  n = "$name"
  z = esc(:($zero))
  return :( $o = add($o, $name = create_or_load_lightobs($n, $z, name = $n, alloc = $nc)) )
end

macro addobs(name, T)
  nc = esc(:(num_confs))
  o = esc(:(obs))
  n = "$name"
  t = esc(:($T))
  return :( $o = add($o, $name = create_or_load_obs($n, $t, name = $n, alloc = $nc)) )
end





# -------------------------------------------------------
#                  Define MeasParams
# -------------------------------------------------------
@with_kw mutable struct MeasParams
  requested::Vector{Symbol} = Symbol[]
  todo::Vector{Symbol} = Symbol[]

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













function create_todolist!(mp::MeasParams)
  mp.todo = Symbol[]

  if !endswith(mp.outfile, ".running") && isfile(mp.outfile*".running")
    # job has finished since last measurement run
    mv(mp.outfile*".running", mp.outfile)
  end


  if isfile(mp.outfile) && !mp.overwrite
    # Start/continue unfinished measurements but don't overwrite anything
    println("Measurement file found.");
    println("Checking what is left todo...")

    allobs = listobs(mp.outfile)
    confs = loadobs_frommemory(mp.dqmc_outfile, "obs/configurations")
    N = length(confs)

    counts = Dict{Symbol, Int64}()

    h5open(mp.outfile, "r") do f
      for o in mp.requested
        o == :binder && continue # special case binder cumulant

        # p = joinpath("obs/", string(o))
        p = joinpath("obj/", string(o))
        @show p

        if !HDF5.has(f, p)
          push!(mp.todo, o)
        else
          # count = read(f[joinpath(p, "count")])
          count = length(read(f[p]))
          @show count
          if count != N
            push!(mp.todo, o)
            counts[o] = count
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
      push!(:binder, mp.todo)
    end

    mp.confs_iterator_start = try unique(values(counts))[1] + 1 catch er 1 end

  else # there is no meas outfile yet or we are in overwrite mode
    mp.todo = copy(mp.requested)
  end

  mp.todo
end



# -------------------------------------------------------
#                        Main
# -------------------------------------------------------
function main(mp::MeasParams)

  create_todolist!(mp)

  if isempty(mp.todo)
    @info "Nothing on the todo list. Exiting."
    exit()
  end

  # Overwrite mode?
  if isfile(mp.outfile) && mp.overwrite
    jldopen(mp.outfile, "r+") do f
      for o in mp.todo
        p = joinpath("obj/", string(o))
        HDF5.has(f.plain, p) && o_delete(f, p)
      end
    end
  end


  # -------------- Load DQMC params/results -----------------
  println("\nLoading DQMC results...")

  greens = nothing; 
  mc = nothing;
  local confs

  try
    confs = loadobs_frommemory(mp.dqmc_outfile, "obs/configurations")
  catch err
    println("Couldn't read configuration data. Probably no configurations yet? Exiting.")
    exit()
  end

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
  num_confs = length(confs)
  nsweeps = num_confs * mp.p.write_every_nth
  obs = NamedTuple() # list of observables to measure

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
    obs = add(obs, binder = Observable(Float64, name="binder"))
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



  # --------------------- Measure -------------------------
  print("\nPrepared Observables ");
  println(keys(obs))
  println()
  reset_timer!(mp.to)
  measure(mp, obs, confs, greens, mc)

  # ------------ Export results   ----------------
  export_results(mp, obs, nsweeps)
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

  # TODO: Be smarter for Observable objects (bosonic observables)

  nothing
end







# -------------------------------------------------------
#                      MEASUREMENTS
# -------------------------------------------------------
function measure(mp::MeasParams, obs::NamedTuple{K,V}, confs, greens, mc) where {K,V}
    # greens = (lazy) observable of greens functions or nothing
    num_confs = length(confs)
    nsweeps = num_confs * mp.p.write_every_nth

    # ----------------- Measure loop ------------------------
    length(obs) > 0 && println("Measuring ...");
    flush(stdout)

    @mytimeit mp.to "measure loop" begin
      @inbounds @views @showprogress for i in mp.confs_iterator_start:num_confs
          println(i); flush(stdout);
          @mytimeit mp.to "load conf" conf = confs[i]

          @mytimeit mp.to "bosonic" measure_bosonic(mp, obs, conf, i)

          @mytimeit mp.to "load etgf" g = need_to_load_etgf(obs) ? greens[i] : nothing # load single greens from disk per MCO.jl
          @mytimeit mp.to "fermionic" measure_fermionic(mp, obs, conf, g, mc, i)


          @mytimeit mp.to "intermediate save" if mod1(i, mp.save_after) == mp.save_after
            save_obs_objects(mp, obs)
          end

          # TODO: save and exit(42) when WLT is reached in next iteration
      end
    end

    @mytimeit mp.to "final save" save_obs_objects(mp.obs)

    if TIMING
      println()
      display(mp.to)
      println()
      flush(stdout)
    end

    # ------------ Postprocessing   ----------------
    if :binder in keys(obs)
        # binder postprocessing
        m2ev2 = mean(obs[:m2s])^2
        m4ev = mean(obs[:m4s])

        push!(obs[:binder], m4ev/m2ev2)
    end
end







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





function measure_fermionic(mp, obs, conf, greens, mc, i)
    if !isnothing(mc)
      mc.p.hsfield = conf
    end
    todo = keys(obs)

    # etpc
    @mytimeit mp.to "etpc" if :etpc_plus in todo
        measure_etpc!(mc, greens)
        push!(obs[:etpc_plus], mc.s.meas.etpc_plus)
        push!(obs[:etpc_minus], mc.s.meas.etpc_minus)
    end

    # etcdc
    @mytimeit mp.to "etcdc" if :etcdc_plus in todo
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
    @mytimeit mp.to "zfpc" if :zfpc_plus in todo
        measure_zfpc!(mc, Gt0)
        push!(obs[:zfpc_plus], mc.s.meas.zfpc_plus)
        push!(obs[:zfpc_minus], mc.s.meas.zfpc_minus)
    end

    # zfcdc
    @mytimeit mp.to "zfcdc" if :zfcdc_plus in todo
        measure_zfcdc!(mc, greens, Gt0, G0t)
        push!(obs[:zfcdc_plus], mc.s.meas.zfcdc_plus)
        push!(obs[:zfcdc_minus], mc.s.meas.zfcdc_minus)
    end

    @mytimeit mp.to "sfdensity" if :sfdensity in todo
        push!(obs[:sfdensity], measure_sfdensity(mc, zfccc))
    end

    # # tdgfs
    # if :tdgfs_Gt0 in keys(obs)        
    #     push!(obs[:tdgfs_Gt0], VectorOfArray(Gt0))
    #     push!(obs[:tdgfs_G0t], VectorOfArray(G0t))
    # end

    nothing
end









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
    h5open(mp.outfile, "r+") do fout
        HDF5.has(fout, "nsweeps") && HDF5.o_delete(fout, "nsweeps")
        HDF5.has(fout, "write_every_nth") && HDF5.o_delete(fout, "write_every_nth")
        fout["nsweeps"] = nsweeps
        fout["write_every_nth"] = mp.p.write_every_nth
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
  if isfile(mp.outfile)
    x = jldopen(mp.outfile, "r") do f
      p = joinpath("obj/", abbrev)
      if HDF5.has(f.plain, p)
        # println("loaded old lightobs")
        read(f[p])
      end
    end
    !isnothing(x) && return x
  end

  # couldn't be loaded -> create new LightObservable
  # println("created new lightobs")
  return LightObservable(args...; kwargs...)
end

"""
Create or load Observable
"""
function create_or_load_obs(abbrev::String, args...; kwargs...)
  if isfile(mp.outfile)
    x = jldopen(mp.outfile, "r") do f
      p = joinpath("obj/", abbrev)
      if HDF5.has(f.plain, p)
        read(f[p])
      end
    end
    !isnothing(x) && return x
  end

  # couldn't be loaded -> create new LightObservable
  return Observable(args...; kwargs...)
end

nothing