using Distributed, MonteCarloObservable

const dqmc_file = joinpath(ENV["JULIA_DQMC"], "app/dqmc.jl")
const params_dir = joinpath(ENV["JULIA_DQMC"], "test/parameters")


function killworkers()
    while nprocs() > 1
        rmprocs(workers()[end])
    end
end


function change_sweeps_in_xml(xml, sweeps)
    str = read(xml, String)
    strs = split(str, "SWEEPS")
    start = findfirst(isequal('>'), strs[2])+1
    stop = findfirst(isequal('<'), strs[2])-1
    cur_sweeps = strs[2][start:stop]
    strs[2] = replace(strs[2], cur_sweeps => string(sweeps), count=1)
    write(xml, join(strs, "SWEEPS"))
    nothing
end



"""
Creates a fresh temporary directory for e.g. dqmc simulations.

To be used as

        intmpdir() do tmp_dir
            # whatever
        end
"""
function intmpdir(f)
    killworkers()
    mktempdir() do tmp_dir
        cd(tmp_dir) do

            # code
            tmp_dir = pwd()
            f(tmp_dir)

        end
    end
    killworkers()
end

"""
Run a dqmc simulation on a worker process. Returns the elapsed seconds.
"""
function run_dqmc(in_xml)
    addprocs(1) # Create worker to run dqmc.jl
    t = @elapsed @sync @spawnat workers()[1] begin
        empty!(ARGS)
        push!(ARGS, in_xml)
        include(dqmc_file)
    end
    killworkers()
    return t
end


"""
Temporarily set an environmental variable to a value
"""
function with_env_var(f, env, val)
    before_val = haskey(ENV, env) ? ENV[env] : nothing
    @info "Setting ENV[\"$env\"] = $val."
    ENV[env] = val
    f()
    if !isnothing(before_val)
        ENV[env] = before_val
    end
end





if !isfile(dqmc_file)
    @warn "dqmc.jl not found. Skipping Resume DQMC tests."
elseif !isdir(params_dir)
    @warn "parameters directory not found. Skipping Resume DQMC tests."
else
    @testset "Resume DQMC" begin


        intmpdir() do tmp_dir
            # cp(joinpath(params_dir, "resumelong.in.xml"), joinpath(tmp_dir, "resumelong.in.xml"))
            # cp(joinpath(params_dir, "resumeshort.in.xml"), joinpath(tmp_dir, "resumeshort.in.xml"))
            cp(joinpath(params_dir, "resumelong.in.xml"), "resumelong.in.xml")
            cp(joinpath(params_dir, "resumeshort.in.xml"), "resumeshort.in.xml")

            # Run resumelong
            println("Starting DQMC (resumelong)")
            tlong = run_dqmc("resumelong.in.xml")
            println("Done (resumelong)")

            # extract minutes and seconds from tlong/2
            # s = Second(floor(Int, tlong/2))
            # d = Dates.canonicalize(Dates.CompoundPeriod(s))
            # minutes = d.periods[1].value
            # seconds = d.periods[2].value


            @testset "Resume (extension run)" begin
                # Run resumeshort
                println("Starting DQMC (resumeshort)")
                run_dqmc("resumeshort.in.xml")
                println("Done (resumeshort)")


                mv("resumeshort.out.h5", "resumeshort.out.h5.running")
                change_sweeps_in_xml("resumeshort.in.xml", 200)


                # Run resumeshort once again
                println("Starting DQMC (resumeshort - again)")
                run_dqmc("resumeshort.in.xml")
                println("Done (resumeshort - again)")


                # Comparison
                clong = ts_flat("resumelong.out.h5", "obs/configurations");
                cshort = ts_flat("resumeshort.out.h5", "obs/configurations");
                l = Int(size(clong, ndims(clong)) / 2)
                @test @views isapprox(clong[:,:,:,1:l], cshort[:,:,:,1:l]) # should trivially pass
                @test isapprox(clong, cshort)

                glong = ts_flat("resumelong.out.h5", "obs/greens");
                gshort = ts_flat("resumeshort.out.h5", "obs/greens");
                @test @views isapprox(glong[:,:,1:l], gshort[:,:,1:l]) # should trivially pass
                @test isapprox(glong, gshort)
            end

            # VERY ERROR PRONE....
            # @testset "Resume after safe exit (WTL)" begin
            #     cp("resumelong.in.xml", "resumewtl.in.xml")

            #     with_env_var("WALLTIMELIMIT", "0-00:00:$(floor(Int, tlong/2))") do
            #         try
            #             println("Starting DQMC (resumetwtl 1)")
            #             run_dqmc("resumewtl.in.xml")
            #             println("Done (resumetwtl 1)")
            #         catch err
            #             if !isa(err, ProcessExitedException)
            #                 throw(err)
            #             end
            #         end

            #         try
            #             println("Starting DQMC (resumetwtl 2)")
            #             run_dqmc("resumewtl.in.xml")
            #             println("Done (resumetwtl 2)")
            #         catch err
            #             if !isa(err, ProcessExitedException)
            #                 throw(err)
            #             end
            #         end

            #         # check that job hasn't finished but did dump something
            #         if isfile("resumewtl.out.h5")
            #             @warn "RESUME WTL: WTL too long, job did finish. Skipping."
            #         elseif !isfile("resumewtl.out.h5.running")
            #             @warn "RESUME WTL: Job didn't do anything? Skipping."
            #         elseif !("configurations" in listobs("resumewtl.out.h5.running"))
            #             @warn "RESUME WTL: Found .running file but no confs. WTL too short. Skipping."
            #         else
            #             c = loadobs_frommemory("resumewtl.out.h5.running", "obs/configurations")
            #             clong = loadobs_frommemory("resumelong.out.h5", "obs/configurations")
            #             @test isapprox(c, clong[:,:,:,1:length(c)])
            #         end
            #     end
            # end
        end # intmpdir

    end # resume testset
end



