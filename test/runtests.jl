using Test
include("../src/dqmc_framework.jl") # to be replaced by using DQMC or similar

function mc_from_inxml(inxml::AbstractString)
    if SUPPRESS_DQMC_INIT_OUTPUT
        old_stdout = stdout
        (rp, wp) = redirect_stdout()
    end

    @assert isfile(inxml)
    p = Params()
    p.output_file = "$(inxml[1:end-7]).out.h5.running"
    xml2parameters!(p, inxml)
    mc = DQMC(p)
    init!(mc)

    if SUPPRESS_DQMC_INIT_OUTPUT
        redirect_stdout(old_stdout)
        close(wp); close(rp);
    end
    return mc
end

function mc_from_inxml(inxml::AbstractString, initconf)
    mc = mc_from_inxml(inxml)
    mc.p.hsfield .= initconf
    return mc
end











const SUPPRESS_DQMC_INIT_OUTPUT = true


@testset "All Tests" begin

    # tests
    include("O3.jl")

    # no interactions
    include("freefermions.jl")

end

nothing