using Test
include("../dqmc_framework.jl") # to be replaced by using DQMC or similar

# tests
include("O3.jl")

# no interactions
include("freefermions.jl")

nothing