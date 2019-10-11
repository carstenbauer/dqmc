@testset "Boson measurements" begin
    copyto!(mc.p.hsfield, randconf)

    @test isapprox(measure_chi_dynamic(mc.p.hsfield), load("data/O3.jld", "chi_dyn"))
    @test isapprox(measure_chi_static(mc.p.hsfield), 12.420575691388407)
    @test all(isapprox.(measure_binder_factors(mc.p.hsfield), (0.7762859807117749, 0.6026199238496421)))
    a,b = measure_op(mc.p.hsfield)
    @test isapprox(a, 0.5080018386458331)
    @test isapprox(b, [0.514548, 0.536541, 0.472916], atol=1e-4)

    qy, qx, ws = get_momenta_and_frequencies(mc)
    @test qy == qx
    @test isapprox(qy, [0.0, 1.5708, 3.14159], atol=1e-4)
    @test isapprox(ws, [0.0, 6.28319, 12.5664, 18.8496, 25.1327, 31.4159], atol=1e-4)

    qy, qx, ws = get_momenta_and_frequencies(mc.p.hsfield)
    @test qy == qx
    @test isapprox(qy, [0.0, 1.5708, 3.14159], atol=1e-4)
    @test isapprox(ws, [0.0, 6.28319, 12.5664, 18.8496, 25.1327, 31.4159], atol=1e-4)
end

















@testset "Fermion measurements" begin
    init!(mc)

    @testset "Helpers" begin
        @testset "Accessing Greens" begin
            sql = reshape(1:16, (4,4))
            N = mc.l.sites
            greens = mc.s.greens[1:2*N,1:2*N] # fake O(2) greens

            f = () -> begin
                for x in 1:4
                    for y in 1:4
                        siteidx(mc, sql, x, y) == sql[y,x] || return false
                    end
                end
                return true
            end
            @test f()

            # PBC
            @test siteidx(mc, sql, 5, 1) == 1
            @test siteidx(mc, sql, 1, 5) == 1
            @test siteidx(mc, sql, 5, 5) == 1
            @test siteidx(mc, sql, 3, 7) == 11
            @test siteidx(mc, sql, 128, 32) == 16

            # greensidx
            f = () -> begin
                for i in 1:N
                    greensidx(N, 1, i) == i || return false
                    greensidx(N, 2, i) == N+i || return false
                    greensidx(N, 3, i) == 2*N+i || return false
                    greensidx(N, 4, i) == 3*N+i || return false
                end
                return true
            end
            @test f()


            # G for O(3) equal to indexing greens
            f = () -> begin
                for i in 1:4*N, j in 1:4*N
                    G(mc, i, j) == mc.s.greens[i, j] || return false
                end
                return true
            end
            @test f()

            # Gtilde for O(3) equal to indexing I - greens
            f = () -> begin
                for i in 1:4*N, j in 1:4*N
                    Gtilde(mc, i, j) == kd(i,j) - mc.s.greens[i, j] || return false
                end
                return true
            end
            @test f()

            @test isapprox(fullG(mc, mc.s.greens), mc.s.greens)
            @test isapprox(fullGtilde(mc, mc.s.greens), I - mc.s.greens)


            mc.p.opdim = 2

            # G for O(2)/O(1)
            f = () -> begin
                for i in 1:2*N, j in 1:2*N
                    isapprox(G(mc, 2*N+i, 2*N+j, greens), conj(greens[i, j])) || return false
                    isapprox(Gtilde(mc, 2*N+i, 2*N+j, greens), kd(2*N+i, 2*N+j) - conj(greens[i, j])) || return false
                end
                return true
            end
            @test f()

            @test isapprox(fullG(mc, greens), I - fullGtilde(mc, greens))

            mc.p.opdim = 3
        end


        @testset "Permute and FFT Greens" begin
            N = mc.l.sites
            L = mc.p.L
            xu, yd, xd, yu = 1, 2, 3, 4 # definition = old order

            flvtrafo = Dict{Int64, Int64}(xu => xu, yd => yu, xd => xd, yu => yd)

            @test isapprox(permute_greens(permute_greens(mc.s.greens)), mc.s.greens)

            gperm = PseudoBlockArray(permute_greens(mc.s.greens), [N,N,N,N], [N,N,N,N])
            g = PseudoBlockArray(mc.s.greens, [N,N,N,N], [N,N,N,N])

            f = () -> begin
                for j in (xu, yd, xd, yu)
                    for i in (xu, yd, xd, yu)
                        isapprox(getblock(gperm, i, j), getblock(g, flvtrafo[i], flvtrafo[j])) || return false
                    end
                end
                return true
            end
            @test f()


            # FFT
            @test isapprox(ifft_greens(mc, fft_greens(mc, mc.s.greens; fftshift=false); ifftshift=false), mc.s.greens)
            @test isapprox(ifft_greens(mc, fft_greens(mc, mc.s.greens; fftshift=true); ifftshift=true), mc.s.greens)

            gk = fft_greens(mc, mc.s.greens)
            @test isapprox(gk, load("data/O3.jld", "greens_fft"))

            qs = collect(range(-pi, pi, length=L+1))[1:end-1]
            qs7 = [-2.6927937030769655, -1.7951958020513104, -0.8975979010256552, 0.0, 0.8975979010256552, 1.7951958020513104, 2.6927937030769655]
            @test all(isapprox.(fftmomenta(7; fftshift=true), qs7))
            @test all(isapprox.(fftmomenta(mc; fftshift=true), qs))
            @test all(isapprox.(fftmomenta(7; fftshift=false), ifftshift(qs7)))
            @test all(isapprox.(fftmomenta(mc; fftshift=false), ifftshift(qs)))
        end
    end

    # @testset "Effective -> Actual GF" begin
    #     # TODO
    # end


    # Calculate ETGF and logdet
    g = effective_greens2greens(mc, mc.s.greens)
    lgdet = calculate_logdet(mc)

    # Calculate TDGFs
    allocate_tdgfs!(mc)
    measure_tdgfs!(mc)
    Gt0 = mc.s.meas.Gt0
    G0t = mc.s.meas.G0t


    @testset "ETGF(s)" begin
        gmeas = measure_greens(mc, mc.p.safe_mult, mc.s.current_slice);
        gmeas2, lgdetmeas = measure_greens_and_logdet(mc, mc.p.safe_mult, mc.s.current_slice);

        @test isapprox(g, gmeas)
        @test isapprox(gmeas, gmeas2)
        @test isapprox(lgdet, lgdetmeas)

        # ETGFs at all time slices
        a = calc_all_greens_explicit(mc);
        b = calc_all_greens(mc);

        @test maximum(maximum.(real.(a .- b))) < 1e-12
        @test maximum(maximum.(imag.(a .- b))) < 1e-12
    end


    @testset "Occupation" begin
        @test isapprox(occupation(mc, g), 0.4216, atol=1e-4)
        @test all(isapprox.(occupations_flv(mc, g), 0.4216, atol=1e-4))
    end


    @testset "TDGFs" begin
        @test isapprox(calc_tdgf_direct(mc, 1), Gt0[1])
        # @test_broken isapprox(calc_tdgf_direct(mc, 4), Gt0[4])


        check_unitarity = (u_stack) -> begin
          for i in 1:length(u_stack)
            U = u_stack[i]
            !isapprox(U * adjoint(U), I) && (return false)
          end
          return true
        end

        @test check_unitarity(mc.s.meas.BT0Inv_u_stack)
        @test check_unitarity(mc.s.meas.BBetaT_u_stack)
        @test check_unitarity(mc.s.meas.BT0_u_stack)
        @test check_unitarity(mc.s.meas.BBetaTInv_u_stack)

        # TODO: Check stacks values?

        @test isapprox(estimate_memory_usage_tdgfs(mc), 1.8, atol=1e-1)
        @test isapprox(memory_usage_tdgfs(mc), 1.8, atol=1e-1)
    end



    @testset "ETPC" begin
        allocate_etpc!(mc)
        @test mc.s.meas.etpc_minus == zero(mc.s.meas.etpc_minus)
        @test mc.s.meas.etpc_plus == zero(mc.s.meas.etpc_plus)

        measure_etpc!(mc, g)
        Pm = mc.s.meas.etpc_minus
        Pp = mc.s.meas.etpc_plus

        @test isreal(Pm)
        @test isreal(Pp)
        @test is_reflection_symmetric(Pm, tol=1e-3)
        @test is_reflection_symmetric(Pp, tol=1e-3)
    end

    @testset "ETCDC" begin
        allocate_etcdc!(mc)
        @test mc.s.meas.etcdc_minus == zero(mc.s.meas.etcdc_minus)
        @test mc.s.meas.etcdc_plus == zero(mc.s.meas.etcdc_plus)

        measure_etcdc!(mc, g)
        Cm = mc.s.meas.etcdc_minus
        Cp = mc.s.meas.etcdc_plus

        @test maximum(imag(Cm)) < 1e-12
        @test maximum(imag(Cp)) < 1e-12
        @test is_reflection_symmetric(Cm, tol=1e-3)
        @test is_reflection_symmetric(Cp, tol=1e-3)
    end

    @testset "ZFPC" begin
        allocate_zfpc!(mc)
        @test mc.s.meas.zfpc_minus == zero(mc.s.meas.zfpc_minus)
        @test mc.s.meas.zfpc_plus == zero(mc.s.meas.zfpc_plus)

        measure_zfpc!(mc, Gt0)
        Pm = mc.s.meas.zfpc_minus
        Pp = mc.s.meas.zfpc_plus

        @test isreal(Pm)
        @test isreal(Pp)
        @test is_reflection_symmetric(Pm, tol=1e-3)
        @test is_reflection_symmetric(Pp, tol=1e-3)

        M = mc.p.slices
        measure_zfpc!(mc, [g for _ in 1:M]) # ~ τ = 0
        @test Pm/M ≈ mc.s.meas.etpc_minus
        @test Pp/M ≈ mc.s.meas.etpc_plus
    end

    @testset "ZFCDC" begin
        allocate_zfcdc!(mc)
        @test mc.s.meas.zfcdc_minus == zero(mc.s.meas.zfcdc_minus)
        @test mc.s.meas.zfcdc_plus == zero(mc.s.meas.zfcdc_plus)

        measure_zfcdc!(mc, g, Gt0, G0t)
        Cm = mc.s.meas.zfcdc_minus
        Cp = mc.s.meas.zfcdc_plus

        @test maximum(imag(Cm)) < 1e-12
        @test maximum(imag(Cp)) < 1e-12
        @test is_reflection_symmetric(Cm, tol=1e-3)
        @test is_reflection_symmetric(Cp, tol=2e-3) # is 2e-3 a problem?
    end


    @testset "ZFCCC + Superfluid density" begin
        allocate_zfccc!(mc)
        @test mc.s.meas.zfccc_xx == zero(mc.s.meas.zfccc_xx)
        @test mc.s.meas.zfccc_yy == zero(mc.s.meas.zfccc_yy)

        measure_zfccc!(mc, g, Gt0, G0t)
        Λxx = mc.s.meas.zfccc_xx
        Λyy = mc.s.meas.zfccc_yy

        @test maximum(imag(Λxx)) < 1e-12
        @test is_reflection_symmetric(Λxx, tol=2e-3) # is 2e-3 a problem?
        @test maximum(imag(Λyy)) < 1e-12
        # TODO: For our model Λxx ≈ Λyy'. perhaps we should test that?
        @test is_reflection_symmetric(Λyy, tol=1e-2) # is 1e-2 a problem?

        @test isapprox(measure_sfdensity(mc, Λxx, Λyy), 0.05157097614282691) # This value is only self-consistent.
    end


    @test isnothing(deallocate_tdgfs_stacks!(mc))
    @test length(mc.s.meas.BT0Inv_u_stack) == 0
    @test length(mc.s.meas.BT0Inv_d_stack) == 0
    @test length(mc.s.meas.BT0Inv_t_stack) == 0
    @test length(mc.s.meas.BBetaT_u_stack) == 0
    @test length(mc.s.meas.BBetaT_d_stack) == 0
    @test length(mc.s.meas.BBetaT_t_stack) == 0
    @test length(mc.s.meas.BT0_u_stack) == 0
    @test length(mc.s.meas.BT0_d_stack) == 0
    @test length(mc.s.meas.BT0_t_stack) == 0
    @test length(mc.s.meas.BBetaTInv_u_stack) == 0
    @test length(mc.s.meas.BBetaTInv_d_stack) == 0
    @test length(mc.s.meas.BBetaTInv_t_stack) == 0

end

nothing
