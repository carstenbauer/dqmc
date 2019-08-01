module BenchLinAlg

using BenchmarkTools
using SparseArrays, LinearAlgebra

include("../src/linalg.jl")

suite = BenchmarkGroup()

dtype = ComplexF64
n = 1000

suite["decompose_udt"] = @benchmarkable(
    decompose_udt(x),
    setup=(x = rand($dtype, $n, $n))
)

end  # module
BenchLinAlg.suite