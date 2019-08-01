# load comparison data
using JLD, LinearAlgebra, SparseArrays
(@isdefined randconf) || (global const randconf = load("data/O3.jld", "randconf"))


# function rewrite()
#     objs = Dict{String, Any}()

#     jldopen("data/O3.jld", "r") do f
#         objkeys = names(f)
#         for k in objkeys
#             objs[k] = read(f[k])
#         end 
#     end

#     jldopen("O3new.jld", "w") do fout
#         for (k, v) in objs
#             fout[k] = v 
#         end
#     end
#     nothing
# end


# set up minimal O3 simulation
mc = mc_from_inxml("parameters/O3_generic_small_system.in.xml", randconf)
mc_odd_L = mc_from_inxml("parameters/O3_odd_L_small_system.in.xml", load("data/O3.jld", "randconf_odd_L"))
mc_nob = mc_from_inxml("parameters/O3_no_bfield_small_system.in.xml", randconf)
mc_nob_nochkr = mc_from_inxml("parameters/O3_no_bfield_no_chkr_small_system.in.xml", randconf)
# set to nothing at the end of this file to trigger GC


@testset "O3 model" begin

    @testset "types" begin
        @test geltype(mc) == ComplexF64
        @test heltype(mc) == ComplexF64
        @test cbtype(mc) == CBAssaad

        @test geltype(mc_nob) == ComplexF64
        @test heltype(mc_nob) == Float64
        @test cbtype(mc_nob) == CBAssaad

        @test geltype(mc_nob_nochkr) == ComplexF64
        @test heltype(mc_nob_nochkr) == Float64
        @test cbtype(mc_nob_nochkr) == CBFalse
    end


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
        @test isapprox(mc.l.hopping_matrix_exp, load("data/O3.jld", "hopping_matrix_exp"))
        @test isapprox(mc.l.hopping_matrix_exp_inv, load("data/O3.jld", "hopping_matrix_exp_inv"))
    end


    # test peirls phases
    # TODO: load a NNNN model for this (to also check higher order phases)
    include("tests_O3_peirls.jl")


    @testset "checkerboard hopping matrices" begin
        # init_checkerboard_matrices(mc_nob)
        @test find_four_site_hopping_corners(mc_nob.l) == ([1, 3, 9, 11], [6, 8, 14, 16])

        @test isapprox(mc_nob.l.chkr_hop[1], load("data/O3.jld", "nob_chkr_hop1"))
        @test isapprox(mc_nob.l.chkr_hop[2], load("data/O3.jld", "nob_chkr_hop2"))
        @test isapprox(mc_nob.l.chkr_hop_inv[1], load("data/O3.jld", "nob_chkr_hop_inv1"))
        @test isapprox(mc_nob.l.chkr_hop_inv[2], load("data/O3.jld", "nob_chkr_hop_inv2"))

        @test isapprox(mc_nob.l.chkr_hop_half[1], load("data/O3.jld", "nob_chkr_hop_half1"))
        @test isapprox(mc_nob.l.chkr_hop_half[2], load("data/O3.jld", "nob_chkr_hop_half2"))
        @test isapprox(mc_nob.l.chkr_hop_half_inv[1], load("data/O3.jld", "nob_chkr_hop_half_inv1"))
        @test isapprox(mc_nob.l.chkr_hop_half_inv[2], load("data/O3.jld", "nob_chkr_hop_half_inv2"))

        @test isapprox(mc_nob.l.chkr_mu, load("data/O3.jld", "nob_chkr_mu"))
        @test isapprox(mc_nob.l.chkr_mu_half, load("data/O3.jld", "nob_chkr_mu_half"))
        @test isapprox(mc_nob.l.chkr_mu_inv, load("data/O3.jld", "nob_chkr_mu_inv"))
        @test isapprox(mc_nob.l.chkr_mu_half_inv, load("data/O3.jld", "nob_chkr_mu_half_inv"))
    end


    @testset "checkerboard hopping matrices (Bfield)" begin
        @test isapprox(build_four_site_hopping_matrix_Bfield(mc, 3, 1, 2), load("data/O3.jld", "build_four_site_hopping_matrix_Bfield"))

        @test isapprox(mc.l.chkr_hop[1], load("data/O3.jld", "chkr_hop1"))
        @test isapprox(mc.l.chkr_hop[2], load("data/O3.jld", "chkr_hop2"))
        @test isapprox(mc.l.chkr_hop_inv[1], load("data/O3.jld", "chkr_hop_inv1"))
        @test isapprox(mc.l.chkr_hop_inv[2], load("data/O3.jld", "chkr_hop_inv2"))

        @test isapprox(mc.l.chkr_hop_half[1], load("data/O3.jld", "chkr_hop_half1"))
        @test isapprox(mc.l.chkr_hop_half[2], load("data/O3.jld", "chkr_hop_half2"))
        @test isapprox(mc.l.chkr_hop_half_inv[1], load("data/O3.jld", "chkr_hop_half_inv1"))
        @test isapprox(mc.l.chkr_hop_half_inv[2], load("data/O3.jld", "chkr_hop_half_inv2"))

        @test isapprox(mc.l.chkr_mu, load("data/O3.jld", "chkr_mu"))
        @test isapprox(mc.l.chkr_mu_half, load("data/O3.jld", "chkr_mu_half"))
        @test isapprox(mc.l.chkr_mu_inv, load("data/O3.jld", "chkr_mu_inv"))
        @test isapprox(mc.l.chkr_mu_half_inv, load("data/O3.jld", "chkr_mu_half_inv"))
    end


    @testset "generic checkerboard hopping matrices" begin
        # TODO: generic checkerboard
    end


    @testset "interactions" begin
        @testset "setblockdiag!" begin
            A = sprand(mc.l.sites*4, mc.l.sites*3, 0.1)
            B = copy(A)
            setblockdiag!(mc.l, B, 2,3, 1:mc.l.sites)
            @test diag(B[(mc.l.sites+1):(2*mc.l.sites),(2*mc.l.sites+1):(3*mc.l.sites)]) == collect(1.0:mc.l.sites)
            setblockdiag!(mc.l, B, 2,3, diag(A[(mc.l.sites+1):(2*mc.l.sites),(2*mc.l.sites+1):(3*mc.l.sites)]))
            @test A == B
        end

        @testset "interaction matrix exponentials" begin
            eV = similar(mc_nob.s.eV)
            interaction_matrix_exp!(mc_nob, 3, 1., eV)
            @test isapprox(eV, load("data/O3.jld", "nob_eVplus"))
            @test isapprox(eV, interaction_matrix_exp(mc_nob, 3, 1.))
            interaction_matrix_exp!(mc_nob, 3, -1., eV)
            @test isapprox(eV, load("data/O3.jld", "nob_eVminus"))

            @test isapprox(interaction_matrix_exp_op(mc_nob, [0.130018, 0.792039, 0.683411], 1.), load("data/O3.jld", "nob_eVexpop"))
        end

        @testset "interaction matrix exponentials (Bfield)" begin
            eV = similar(mc.s.eV)
            interaction_matrix_exp!(mc, 3, 1., eV)
            @test isapprox(eV, load("data/O3.jld", "eVplus"))
            @test isapprox(eV, interaction_matrix_exp(mc, 3, 1.))
            interaction_matrix_exp!(mc, 3, -1., eV)
            @test isapprox(eV, load("data/O3.jld", "eVminus"))

            @test isapprox(interaction_matrix_exp_op(mc, [0.130018, 0.792039, 0.683411], 1.), load("data/O3.jld", "eVexpop"))
        end
    end


    @testset "local updates" begin
        init!(mc)
        @test isapprox(calc_detratio(mc, 7, [0.488033, 0.0196912, 0.438309]), 1.000380293015979 + 1.0842021724855044e-18im)
        @test isapprox(mc.g.delta_i, load("data/O3.jld", "delta_i"))
        @test isapprox(mc.g.M, load("data/O3.jld", "M"))

        update_greens!(mc, 7)
        @test isapprox(mc.g.greens, load("data/O3.jld", "afterupdate_greens"))

        # TODO: test full local update
        Random.seed!(123456789) # set seed
        @test local_updates(mc) == 0.6875 # acc. rate
        @test isapprox(mc.g.greens, load("data/O3.jld", "afterlocal_greens"))
        @test isapprox(mc.p.hsfield, load("data/O3.jld", "afterlocal_hsfield"))
        @test isapprox(mc.p.boson_action, 75.18407422927604)
    end


    @testset "global updates" begin
        # TODO
        # Random.seed!(123456789)
        # hsfield = copy(mc.p.hsfield)
        # global_update_perform_shift!(mc)
        # randsite = rand(1:mc.l.sites)
        # randslice = rand(1:mc.p.slices)
        # p.hsfield[:,randsite,randslice]
        # isapprox( == )

        # TODO: test global_update_backup_swap!
    end


    @testset "stack" begin
        init!(mc)
        @test isapprox(mc.g.greens, calc_greens(mc, mc.g.current_slice)) # compare against "exact" greens

        @testset "greens (+ logdet) calculation" begin
            # load Ur, Dr, Tr, Ul, Dl, Tl of time slice 1, direction up
            mc.g.Ur = load("data/O3.jld", "Ur")
            mc.g.Dr = load("data/O3.jld", "Dr")
            mc.g.Tr = load("data/O3.jld", "Tr")
            mc.g.Ul = load("data/O3.jld", "Ul")
            mc.g.Dl = load("data/O3.jld", "Dl")
            mc.g.Tl = load("data/O3.jld", "Tl")
            greens = load("data/O3.jld", "greens")
            calculate_greens(mc)
            ld = calculate_logdet(mc)
            gfresh, ldfresh = calc_greens_and_logdet(mc, 1)
            @test isapprox(mc.g.greens, greens) # compare against dumped greens
            @test isapprox(mc.g.greens, gfresh) # compare against "exact" freshly calculated greens
            @test isapprox(ld, ldfresh) # compare against "exact" logdet
        end

        @testset "wrapping" begin
            g = calc_greens(mc,2)
            gup = wrap_greens(mc, g, 2, 1)
            gdwn = wrap_greens(mc, g, 2, -1)
            @test isapprox(gup, calc_greens(mc, 3))
            @test isapprox(gdwn, calc_greens(mc, 1))
        end

        init!(mc)
        @testset "propagation" begin
            @assert mc.g.current_slice == mc.p.slices
            while mc.g.current_slice != 1
                propagate(mc)
            end
            propagate(mc) # we want to go in direction up
            greens = load("data/O3.jld", "greens")
            @test isapprox(mc.g.greens, greens) # compare against dumped greens
        end

        init!(mc)
        @testset "propagation: greens always within bounds" begin
            @assert mc.g.current_slice == mc.p.slices
            @assert mc.g.direction == -1

            propagate(mc)
            while !(mc.g.current_slice == mc.p.slices && mc.g.direction == -1)
                mc.g.greens
                greens = calc_greens(mc, mc.g.current_slice)
                @test maximum(absdiff(greens, mc.g.greens)) < 1e-12
                propagate(mc)
            end
        end
    end


    @testset "lattice" begin
        @test mc.l.neighbors == [2 3 4 1 6 7 8 5 10 11 12 9 14 15 16 13;
                                5 6 7 8 9 10 11 12 13 14 15 16 1 2 3 4;
                                4 1 2 3 8 5 6 7 12 9 10 11 16 13 14 15;
                                13 14 15 16 1 2 3 4 5 6 7 8 9 10 11 12]
        @test mc.l.time_neighbors == [2 3 4 5 6 7 8 9 10 1; 10 1 2 3 4 5 6 7 8 9]
        @test mc.l.bonds == [1 2 0 0; 1 5 0 0; 2 3 0 0; 2 6 0 0; 3 4 0 0; 3 7 0 0;
                        4 1 0 0; 4 8 0 0; 5 6 0 0; 5 9 0 0; 6 7 0 0; 6 10 0 0;
                        7 8 0 0; 7 11 0 0; 8 5 0 0; 8 12 0 0; 9 10 0 0; 9 13 0 0;
                        10 11 0 0; 10 14 0 0; 11 12 0 0; 11 15 0 0; 12 9 0 0;
                        12 16 0 0; 13 14 0 0; 13 1 0 0; 14 15 0 0; 14 2 0 0;
                        15 16 0 0; 15 3 0 0; 16 13 0 0; 16 4 0 0]
        @test mc.l.dim == 2
    end


    @testset "slice matrices" begin
        init!(mc)
        init!(mc_nob_nochkr)
        init!(mc_odd_L)
        
        get_mc_and_suffix = (chkr) -> begin
            if chkr === CBAssaad
                return mc, ""
            elseif chkr === CBFalse
                return mc_nob_nochkr, "_CBFalse"
            else
                return mc_odd_L, "_CBGeneric"
            end
        end

        for chkr in [CBAssaad, CBFalse, CBGeneric]
            mc, suffix = get_mc_and_suffix(chkr)

            B = slice_matrix(mc, 3, 1.)
            Binv = slice_matrix(mc, 3, -1.)
            @test isapprox(B, load("data/O3.jld", "Bplus"*suffix))
            @test isapprox(Binv, load("data/O3.jld", "Bminus"*suffix))

            # multiply_daggered_B_left!
            A = rand!(similar(mc.g.greens))
            Ares = adjoint(B) * A
            multiply_daggered_B_left!(mc, 3, A)
            @test isapprox(A, Ares)

            # Binv * B * A == A and A * B * Binv == A
            A = rand!(similar(mc.g.greens))
            Aorig = copy(A)
            multiply_B_left!(mc, 3, A)
            multiply_B_inv_left!(mc, 3, A)
            @test isapprox(A, Aorig)
            multiply_B_right!(mc, 3, A)
            multiply_B_inv_right!(mc, 3, A)
            @test isapprox(A, Aorig)

            # B * A * Binv == A and Binv * A * B == A
            A = Matrix{geltype(mc)}(I, size(mc.g.greens)...)
            Aorig = copy(A)
            multiply_B_left!(mc, 3, A)
            multiply_B_inv_right!(mc, 3, A)
            @test isapprox(A, Aorig)
            multiply_B_right!(mc, 3, A)
            multiply_B_inv_left!(mc, 3, A)
            @test isapprox(A, Aorig)
        end
    end


    @testset "parameters constructors" begin
        # TODO: check if fields are set properly
    end


    @testset "xml read write" begin
        # TODO: xml read write
    end

    @testset "Measurements" begin
        include("tests_O3_measurements.jl")
    end


end # O3 model


mc = nothing
mc_odd_L = nothing
mc_nob = nothing
mc_nob_nochkr = nothing


nothing