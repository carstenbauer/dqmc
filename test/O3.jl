# load comparison data
using JLD
global const randconf = load("O3.jld", "randconf")


function mc_from_inxml(inxml::AbstractString)
    @assert isfile(inxml)
    p = Params()
    p.output_file = "$(inxml[1:end-7]).out.h5.running"
    xml2parameters!(p, inxml)
    mc = DQMC(p)
    init!(mc)

    mc.p.hsfield .= randconf
    mc
end

# set up minimal O3 simulation
mc = mc_from_inxml("parameters/O3_generic_small_system.in.xml")
mc_nob = mc_from_inxml("parameters/O3_no_bfield_small_system.in.xml")
mc_nob_nochkr = mc_from_inxml("parameters/O3_no_bfield_no_chkr_small_system.in.xml")




@testset "O3 model" begin

    @testset "boson action" begin
        @test isapprox(calc_boson_action(mc, randconf), 75.57712964980982)
        @test isapprox(calc_boson_action_diff(mc, 3, 10, [0.370422, 0.797014, 0.956094]), 0.015453643701304112)

        # edrun test
        mc.p.edrun = true
        @test isapprox(calc_boson_action(mc, randconf), 16.348917129437076)
        @test isapprox(calc_boson_action_diff(mc, 3, 10, [0.370422, 0.797014, 0.956094]), 0.014491910524600016)        
        mc.p.edrun = false
    end

    @testset "hopping matrices" begin
        # init_hopping_matrices(mc)
        @test isapprox(mc.l.hopping_matrix_exp, load("O3.jld", "hopping_matrix_exp"))
        @test isapprox(mc.l.hopping_matrix_exp_inv, load("O3.jld", "hopping_matrix_exp_inv"))
    end

    @testset "checkerboard hopping matrices" begin
        # init_checkerboard_matrices(mc_nob)
        @test find_four_site_hopping_corners(mc_nob.l) == ([1, 3, 9, 11], [6, 8, 14, 16])

        @test isapprox(mc_nob.l.chkr_hop[1], load("O3.jld", "nob_chkr_hop1"))
        @test isapprox(mc_nob.l.chkr_hop[2], load("O3.jld", "nob_chkr_hop2"))
        @test isapprox(mc_nob.l.chkr_hop_inv[1], load("O3.jld", "nob_chkr_hop_inv1"))
        @test isapprox(mc_nob.l.chkr_hop_inv[2], load("O3.jld", "nob_chkr_hop_inv2"))

        @test isapprox(mc_nob.l.chkr_hop_half[1], load("O3.jld", "nob_chkr_hop_half1"))
        @test isapprox(mc_nob.l.chkr_hop_half[2], load("O3.jld", "nob_chkr_hop_half2"))
        @test isapprox(mc_nob.l.chkr_hop_half_inv[1], load("O3.jld", "nob_chkr_hop_half_inv1"))
        @test isapprox(mc_nob.l.chkr_hop_half_inv[2], load("O3.jld", "nob_chkr_hop_half_inv2"))

        @test isapprox(mc_nob.l.chkr_mu, load("O3.jld", "nob_chkr_mu"))
        @test isapprox(mc_nob.l.chkr_mu_half, load("O3.jld", "nob_chkr_mu_half"))
        @test isapprox(mc_nob.l.chkr_mu_inv, load("O3.jld", "nob_chkr_mu_inv"))
        @test isapprox(mc_nob.l.chkr_mu_half_inv, load("O3.jld", "nob_chkr_mu_half_inv"))
    end


    @testset "checkerboard hopping matrices (Bfield)" begin
        @test isapprox(build_four_site_hopping_matrix_Bfield(mc, 3, 1, 2), load("O3.jld", "build_four_site_hopping_matrix_Bfield"))

        # TODO: to be continued...
    end

    # @testset "interaction matrix" begin
    #     @test 
    # end

end # O3 model