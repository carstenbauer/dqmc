using Distributed, MonteCarloObservable

dqmc_file = joinpath(ENV["JULIA_DQMC"], "app/dqmc.jl")
params_dir = joinpath(ENV["JULIA_DQMC"], "test/parameters")


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


if !isfile(dqmc_file)
    @warn "dqmc.jl not found. Skipping Resume DQMC tests."
elseif !isdir(params_dir)
    @warn "parameters directory not found. Skipping Resume DQMC tests."
else

    @testset "Resume DQMC" begin
        killworkers() # to be safe

        mktempdir() do tmp_dir
            cd(tmp_dir) do
                @show tmp_dir
                cp(joinpath(params_dir, "resumelong.in.xml"), joinpath(tmp_dir, "resumelong.in.xml"))
                cp(joinpath(params_dir, "resumeshort.in.xml"), joinpath(tmp_dir, "resumeshort.in.xml"))

                # Run resumelong
                println("Starting DQMC (resumelong)")
                addprocs(1) # Create worker to run dqmc.jl
                @sync @spawnat workers()[1] begin
                    empty!(ARGS)
                    push!(ARGS, "resumelong.in.xml")
                    include(dqmc_file)
                end
                println("Done (resumelong)")


                killworkers()


                # Run resumeshort
                println("Starting DQMC (resumeshort)")
                addprocs(1)
                @sync @spawnat workers()[1] begin
                    empty!(ARGS)
                    push!(ARGS, "resumeshort.in.xml")
                    include(dqmc_file)
                end
                println("Done (resumeshort)")


                killworkers()


                mv("resumeshort.out.h5", "resumeshort.out.h5.running")
                change_sweeps_in_xml("resumeshort.in.xml", 200)


                # Run resumeshort once again
                println("Starting DQMC (resumeshort - again)")
                addprocs(1)
                @sync @spawnat workers()[1] begin
                    empty!(ARGS)
                    push!(ARGS, "resumeshort.in.xml")
                    include(dqmc_file)
                end
                println("Done (resumeshort - again)")


                # Comparison
                clong = ts_flat("resumelong.out.h5", "obs/configurations");
                cshort = ts_flat("resumeshort.out.h5", "obs/configurations");
                l = Int(size(clong, ndims(clong)) / 2)
                @test @views isapprox(clong[:,:,:,1:l], cshort[:,:,:,1:l])
                @test isapprox(clong, cshort)

                glong = ts_flat("resumelong.out.h5", "obs/greens");
                gshort = ts_flat("resumeshort.out.h5", "obs/greens");
                @test @views isapprox(glong[:,:,1:l], gshort[:,:,1:l])
                @test isapprox(glong, gshort)

                killworkers()
            end
        end
    end

end