@testset "Boson measurements" begin
    # TODO
end

















@testset "Fermion measurements" begin
    # TODO: ETGF
    # TODO: ETPC etc.


    @testset "TDGF" begin

        let mc = mc_from_inxml("parameters/O3_noninteracting_L_10_safe_mult_1.in.xml")
            allocate_tdgf!(mc)
            Gt0 = mc.s.meas.Gt0
            G0t = mc.s.meas.G0t

            @assert mc.p.safe_mult == 1
            # meas = mc.s.meas
            # N = mc.l.sites
            # L = mc.p.L
            # beta = mc.p.beta
            # flv = 2;
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
                    should_be_zero < 1e-10 || (return false) # TODO: switch to @test when this is working!
                end
                return true
            end
            @test_broken t()

        end
    end

end

nothing