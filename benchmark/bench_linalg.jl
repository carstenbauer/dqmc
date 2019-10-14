module BenchLinAlg

using BenchmarkTools
using SparseArrays, LinearAlgebra

# include("../src/linalg.jl")
# include(joinpath(@__DIR__, "../test/setup_dqmc.jl"))
using QMC

suite = BenchmarkGroup()

mc = mc_from_inxml(joinpath(@__DIR__, "../test/parameters/O3_generic_small_system.in.xml"))
dtype = ComplexF64
n = size(mc.s.greens,1)
A, B, C = rand(dtype, n,n), rand(dtype, n,n), rand(dtype, n,n)
S = sparse(Diagonal(rand(dtype, n)))

U1,D1,T1 = decompose_udt(mc.s.greens)
U2,D2,T2 = decompose_udt(mc.s.greens)
U3,D3,T3 = decompose_udt(mc.s.greens)
res = copy(mc.s.greens)

suite["decompose_udt"] = @benchmarkable decompose_udt($A)
suite["decompose_udt!"] = @benchmarkable decompose_udt!($A, $D1)

suite["Matrix*Sparse (mul!)"] = @benchmarkable mul!($C, $A, $S)
suite["Sparse*Matrix (mul!)"] = @benchmarkable mul!($C, $S, $A)

suite["inv_udt"] = @benchmarkable inv_udt($U1, $D1, $T1)
suite["inv_udt!"] = @benchmarkable inv_udt!($C, $U1, $D1, $T1)

suite["inv_one_plus_two_udts!"] = @benchmarkable inv_one_plus_two_udts!($mc,$U1,$D1,$T1,$U2,$D2,$T2,$U3,$D3,$T3)
suite["inv_one_plus_two_udts!"] = @benchmarkable inv_one_plus_two_udts!($mc,$res,$U2,$D2,$T2,$U3,$D3,$T3)
suite["inv_one_plus_udt_scalettar!"] = @benchmarkable inv_one_plus_udt_scalettar!($mc,$res,$U3,$D3,$T3)
suite["inv_sum_udts_scalettar!"] = @benchmarkable inv_sum_udts_scalettar!($mc,$res,$U1,$D1,$T1,$U2,$D2,$T2)

end  # module
BenchLinAlg.suite
