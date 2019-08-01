println("Running tests on ", gethostname(), ".")
using Test, Random, KrylovKit, BlockArrays, MonteCarloObservable

include("setup_dqmc.jl")


@testset "All Tests" begin

    # General tests
    include("tests_linalg.jl")
    include("tests_statistics.jl")

    # O3 model
    include("tests_O3.jl")

    # no interactions
    include("tests_freefermions.jl")

    # Resume
    # include("tests_resume.jl")

end

nothing