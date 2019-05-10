println("Running tests on ", gethostname(), ".")
using Test, Random, KrylovKit, BlockArrays, MonteCarloObservable
include("../src/dqmc_framework.jl") # to be replaced by using DQMC or similar

function mc_from_inxml(inxml::AbstractString)
    Random.seed!(1234)
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


if !haskey(ENV, "JULIA_DQMC")
    ENV["JULIA_DQMC"] = dirname(dirname(@__FILE__)) # parent directory
end







(@isdefined SUPPRESS_DQMC_INIT_OUTPUT) || (const SUPPRESS_DQMC_INIT_OUTPUT = false)


@testset "All Tests" begin

    # General tests
    include("tests_linalg.jl")
    include("tests_statistics.jl")

    # O3 model
    include("tests_O3.jl")

    # no interactions
    include("tests_freefermions.jl")

    # Resume
    include("tests_resume.jl")

end

nothing