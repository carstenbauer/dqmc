using SparseArrays, LinearAlgebra, Arpack
include("deps/ed.jl")


"""
    perform_ed(beta = 8., nmax::Integer = 22) -> g, nbar

Perform ED for L=2 system and return ETGF and occupation number.

`nmax` specifies how many low energy state we keep in the ED.
"""
function perform_ed(; beta::Float64 = 8., nmax::Integer = 22)
    ns = 8
    params = Dict("txh"=>1.0, "txv"=>0.5, "tyh"=>-0.5, "tyv"=>-1.0, "mu"=>-0.5,
        "r"=>1.0, "lambdax"=>0.0, "lambday"=>0.0, "lambdaz"=>0.0, "lambda0"=>0.0)

    cup, cdn = fermiops(ns);
    cs = vcat(cup, cdn);

    @assert check_fermiops(cup)
    @assert check_fermiops(cdn)

    H = generate_H_SDW(params, ns, cup, cdn);

    evals, evecs, nconverged = eigs(H, nev=nmax+1, which=:SR); # eigenstates are columns of evecs
    idcs = findall(x -> abs(x) < 1e-15, evecs)
    evecs[idcs] .= 0.;

    # calc ETGF
    g = GF(cs, evecs, evals, beta);

    # calc occupation
    n = spzeros(size(cs[1])...)
    for c in cs
        n = n + c' * c
    end
    nbar = EV(n, evecs, evals, beta)

    #@assert isapprox(real(sum(1 .- diag(g))), nbar)

    return g, nbar
end






@testset "Compare Greens to ED" begin
    mc = mc_from_inxml("parameters/free_L_2_B_8.in.xml")
    ged, nbared = perform_ed(beta = 8.0)

    @test isreal(mc.s.greens)
    @test maximum(abs.(mc.s.greens - ged[1:8,1:8])) < 1e-6
    @test abs(real(sum(1 .- diag(mc.s.greens))) - nbared/2) < 1e-6
end




nothing