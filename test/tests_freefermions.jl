using SparseArrays, LinearAlgebra, Arpack
include("deps/ed.jl")


# -----------------------------------------------------
#              Real-space ED (free)
# -----------------------------------------------------
"""
    perform_ed(beta = 8., nmax::Integer = 22) -> obs

Perform ED for L=2 system in real-space.

Returns a couple of observable `obs` as a `Dict{String, Any}`.

`nmax` specifies how many low energy state we keep in the ED.
"""
function perform_ed(; beta::Float64 = 8., nmax::Integer = 22)
    println("Performing real-space ED..."); flush(stdout);
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

    obs = Dict{String, Any}()

    # calc ETGF
    g = GF(cs, evecs, evals, beta);
    obs["greens"] = real(g)

    # calc occupation
    n = spzeros(size(cs[1])...)
    for c in cs
        n = n + c' * c
    end
    obs["occupation"] = EV(n, evecs, evals, beta)
    
    # calc ETPC, ETCDC
    obs["etpc_minus"], obs["etpc_plus"] = etpc(cs, evecs, evals, beta)
    obs["etcdc_minus"], obs["etcdc_plus"] = etcdc(cs, evecs, evals, beta)
    
    #@assert isapprox(real(sum(1 .- diag(g))), nbar)

    println("Done with ED."); flush(stdout);

    return obs
end






@testset "Compare to real-space ED (free)" begin
    mc_ed = mc_from_inxml("parameters/free_L_2_B_8.in.xml")
    ed = perform_ed(beta = 8.0)

    L, N = 2, 4
    flv = 2

    # ETGF must be real
    @test isreal(mc_ed.s.greens)
    g = real(mc_ed.s.greens)
    
    # Compare ETGF (not exactly because real-space ED isn't exact)
    @test maximum(g - ed["greens"][1:8, 1:8]) < 1e-6
    
    @test abs(occupation(mc_ed)*N*flv - ed["occupation"]/2) < 1e-6
    
    # Compare ETPC
    allocate_etpc!(mc_ed)
    etpc!(mc_ed, g)
    @test isapprox(mc_ed.s.meas.etpc_minus, ed["etpc_minus"], atol=1e-6)
    @test isapprox(mc_ed.s.meas.etpc_plus, ed["etpc_plus"], atol=1e-6)
    
    # Compare ETCDC
    allocate_etcdc!(mc_ed)
    etcdc!(mc_ed, g)
    @test isapprox(mc_ed.s.meas.etcdc_minus, ed["etcdc_minus"], atol=1e-6)
    @test isapprox(mc_ed.s.meas.etcdc_plus, ed["etcdc_plus"], atol=1e-6)
end














# -----------------------------------------------------
#              Momentum-space ED (free)
# -----------------------------------------------------
function concat_greens_k_space(gkx, gky)
    Gkx = Diagonal(gkx[:])
    Gky = Diagonal(gky[:])
    # gk = cat(Gkx, Gky, Gkx, Gky, dims=(1,2)); # Full 4Nx4N momentum-space Green's function
    gk = cat(Gkx, Gky, dims=(1,2)); # 2Nx2N momentum-space Green's function
end

"""
    perform_ed_k_space(L = 4, beta = 8.) -> obs

Perform ED for L=2 system in momentum-space single mode basis.

Returns a couple of observable `obs` as a `Dict{String, Any}`.
"""
function perform_ed_k_space(; L::Integer = 8, beta::Float64 = 10.)
    N = L^2
    params_x = Dict("th" => 1.0, "tv" => 0.5, "mu" => -0.5)
    params_y = Dict("th" => -0.5, "tv" => -1.0, "mu" => -0.5)
    obs = Dict{String, Any}()

    # ETGF
    gkx = ifftshift(GF_k_space(params_x, L, beta));
    gky = ifftshift(GF_k_space(params_y, L, beta));

    gk = concat_greens_k_space(gkx, gky); # Full momentum-space Green's function

    g = ifft_greens(gk, flv=2)
    @assert maximum(imag(g)) < 1e-12
    obs["greens"] = real(g) # Full (4xN, 4xN) real-space Green's function



    # TDGF
    taus = collect(0.0:0.1:beta)[1:end-1]

    # Gt0
    gt0_k_x = [TDGF_k_space(params_x, L, beta, tau) for tau in taus]
    gt0_k_y = [TDGF_k_space(params_y, L, beta, tau) for tau in taus]
    gt0_k = concat_greens_k_space.(gt0_k_x, gt0_k_y)
    obs["Gt0"] = real.(ifft_greens.(gt0_k; ifftshift=true, flv=2));
    
    # G0t
    g0t_k_x = [TDGF_k_space(params_x, L, beta, tau; G0t=true) for tau in taus]
    g0t_k_y = [TDGF_k_space(params_y, L, beta, tau; G0t=true) for tau in taus]
    g0t_k = concat_greens_k_space.(g0t_k_x, g0t_k_y)
    obs["G0t"] = real.(ifft_greens.(g0t_k; ifftshift=true, flv=2));

    return obs
end









@testset "Compare to momentum-space ED (free)" begin
    mc_edk = mc_from_inxml("parameters/free_L_8_B_10.in.xml")
    edk = perform_ed_k_space(L = 8, beta = 10.)

    L = mc_edk.p.L
    N = mc_edk.l.sites
    M = mc_edk.p.slices

    # Calculate TDGFs: Gt0, G0t
    allocate_tdgfs!(mc_edk)
    calc_tdgfs!(mc_edk)
    Gt0 = mc_edk.s.meas.Gt0
    G0t = mc_edk.s.meas.G0t

    
    @testset "ETGF" begin
        # ETGF
        @test isapprox(mc_edk.s.greens, edk["greens"])
    end


    @testset "TDGF" begin
        # --------------- DQMC self-consistency checks -----------
        # TDGF == ETGF consistency
        @test isapprox(Gt0[1], mc_edk.s.greens)

        # TDGF: tau = 0 difference btw Gt0 and G0t
        @test isapprox(Gt0[1] - G0t[1], I)

        # TDGF: G(tau,0) == -G(0,beta-tau)
        should_be_zeros = Vector{Float64}(undef, length(2:M))
        t = () -> begin
            for (i, tau) in enumerate(2:M)
                should_be_zero = maximum(Gt0[tau] + G0t[end-(tau-2)])
                should_be_zeros[i] = should_be_zero
                should_be_zero < 1e-10 || (return false)
            end
            return true
        end
        @test t()

        # TDGF: Check realness of FFTs
        @test maximum(maximum.(imag.(fft_greens.(Gt0, flv=2)))) < 1e-12
        @test maximum(maximum.(imag.(fft_greens.(G0t, flv=2)))) < 1e-12
        
        
        # --------------- DQMC vs ED -----------        
        # Gt0
        f = () -> begin
            for i in 1:M
                isapprox(Gt0[i], edk["Gt0"][i]) || return false
            end
            return true
        end
        @test f()
      
        # G0t
        f = () -> begin
            for i in 1:M
                isapprox(G0t[i], edk["G0t"][i]) || return false
            end
            return true
        end
        @test f()
    end



    @testset "ETPC" begin
        allocate_etpc!(mc_edk)
        etpc!(mc_edk, mc_edk.s.greens)
        Pm = mc_edk.s.meas.etpc_minus
        Pp = mc_edk.s.meas.etpc_plus

        @test isreal(Pm)
        @test isreal(Pp)
        @test is_reflection_symmetric(Pm)
        @test is_reflection_symmetric(Pp)
    end

    @testset "ETCDC" begin
        allocate_etcdc!(mc_edk)
        etcdc!(mc_edk, mc_edk.s.greens)
        Cm = mc_edk.s.meas.etcdc_minus
        Cp = mc_edk.s.meas.etcdc_plus

        @test isreal(Cm)
        @test isreal(Cp)
        @test is_reflection_symmetric(Cm)
        @test is_reflection_symmetric(Cp)
    end

    @testset "ZFPC" begin
        allocate_zfpc!(mc_edk)
        zfpc!(mc_edk, Gt0)
        Pm = mc_edk.s.meas.zfpc_minus
        Pp = mc_edk.s.meas.zfpc_plus

        @test isreal(Pm)
        @test isreal(Pp)
        @test is_reflection_symmetric(Pm)
        @test is_reflection_symmetric(Pp)
    end

    @testset "ZFCDC" begin
        allocate_zfcdc!(mc_edk)
        zfcdc!(mc_edk, mc_edk.s.greens, Gt0, G0t)
        Cm = mc_edk.s.meas.zfcdc_minus
        Cp = mc_edk.s.meas.zfcdc_plus

        @test isreal(Cm)
        @test isreal(Cp)
        @test is_reflection_symmetric(Cm)
        @test is_reflection_symmetric(Cp)
    end



    @testset "ZFCCC + Superfluid density" begin
        allocate_zfccc!(mc_edk)
        zfccc!(mc_edk, mc_edk.s.greens, Gt0, G0t)
        Λ = mc_edk.s.meas.zfccc

        @test isreal(Λ)
        @test is_reflection_symmetric(Λ)

        @test isapprox(sfdensity(mc_edk, Λ), 0.2095159287034012) # This value is only self-consistent.
    end
    
end



nothing