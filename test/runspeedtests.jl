using Test, BenchmarkTools
include("../src/dqmc_framework.jl") # to be replaced by using DQMC or similar



@testset "All Speed Tests" begin

    # Linear algebra
    include("speedtests_linalg.jl")

end

nothing