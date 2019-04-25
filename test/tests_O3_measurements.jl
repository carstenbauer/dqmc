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

    # TODO: ETGF
    # TODO: ETPC etc.


    @testset "TDGF" begin

        let mc = mc_from_inxml("parameters/O3_noninteracting_L_10_safe_mult_1.in.xml")
            @assert mc.p.safe_mult == 1
            @assert mc.p.beta == 40
            @assert mc.p.slices == 400
            @assert mc.p.L == 10
            @assert !mc.p.Bfield
            @assert !mc.p.chkr

            allocate_tdgfs!(mc)
            Gt0 = mc.s.meas.Gt0
            G0t = mc.s.meas.G0t
            calc_tdgfs!(mc)



            # TDGF == ETGF consistency
            @test isapprox(Gt0[1], mc.s.greens)

            # TDGF: tau = 0 difference btw Gt0 and G0t
            @test isapprox(Gt0[1] - G0t[1], I)

            # TDGF: G(tau,0) == -G(0,beta-tau)
            should_be_zeros = Vector{Float64}(undef, length(2:mc.p.slices))
            t = () -> begin
                for (i, tau) in enumerate(2:mc.p.slices)
                    should_be_zero = maximum(Gt0[tau] + G0t[end-(tau-2)])
                    should_be_zeros[i] = should_be_zero
                    should_be_zero < 1e-10 || (return false)
                end
                return true
            end
            @test t()


            # TDGF: FFT should be all real without interactions
            function max_imag_of_fft(greens) # expects greens in (N,N) form (single flavor)
                @assert size(greens, 1) == size(greens, 2)
                N = size(greens, 1)
                L = Int(sqrt(N))

                g = reshape(greens, (L,L,L,L))
                gk = ifft( fft(g, (1,2)), (3,4))
                gk = reshape(gk, (N,N))

                return maximum(imag(gk))
            end
            # will select first flavor (1:N, 1:N) sector
            max_imag_of_ffts(multiple_greens, N) = max_imag_of_fft.(getindex.(multiple_greens, Ref(1:N), Ref(1:N)))

            @test maximum(max_imag_of_ffts(Gt0, mc.l.sites)) < 1e-12
            @test maximum(max_imag_of_ffts(G0t, mc.l.sites)) < 1e-12
        end # let

    end # TDGF testset

end

nothing