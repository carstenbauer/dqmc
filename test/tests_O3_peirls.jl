@testset "peirls phases" begin
    println("Checking Peirls phases")
    init!(mc)
    # check peirls phases result
    @test isapprox(filter(x->!isnan(x), mc.l.peirls[1,1]), filter(x->!isnan(x), load("data/O3.jld", "peirls11")))
    @test isapprox(filter(x->!isnan(x), mc.l.peirls[1,2]), filter(x->!isnan(x), load("data/O3.jld", "peirls12")))
    @test isapprox(filter(x->!isnan(x), mc.l.peirls[2,1]), filter(x->!isnan(x), load("data/O3.jld", "peirls21")))
    @test isapprox(filter(x->!isnan(x), mc.l.peirls[2,2]), filter(x->!isnan(x), load("data/O3.jld", "peirls22")))

    p = mc.p
    l = mc.l
    L = mc.l.L
    phis = Matrix{Array{Float64}}(undef,2,2) # colidx = flavor, rowidx = spin 
    for f in 1:2
      for s in 1:2
        phis[s,f] = permutedims(reshape(l.peirls[s,f], (L,L,L,L)), [2,1,4,3])
        B = zeros(2,2) # colidx = flavor, rowidx = spin up,down
        B[1,1] = B[2,2] = 2 * pi / l.sites
        B[1,2] = B[2,1] = - 2 * pi / l.sites

        # Flux check
        fluxes_nn = zeros(ComplexF64, L, L)
        fluxes_nnn_1 = zeros(ComplexF64, L, L)
        fluxes_nnn_2 = zeros(ComplexF64, L, L)
        fluxes_nnn_3 = zeros(ComplexF64, L, L)
        fluxes_nnn_4 = zeros(ComplexF64, L, L)
        fluxes_nnnn = zeros(ComplexF64, L, L)
        fluxes_mixed_1 = zeros(ComplexF64, L, L)
        fluxes_mixed_2 = zeros(ComplexF64, L, L)

        for x in 1:L
            for y in 1:L
                xp = mod1((x + 1),L)
                yp = mod1((y + 1),L)
                ym = mod1((y - 1),L)
                xm = mod1((x - 1),L)
                xp2 = mod1((x + 2),L)
                yp2 = mod1((y + 2),L)
                
                #### Nearest neighbor square
                # up, right, down, left
                flux = phis[s,f][x, y, xp, y] + phis[s,f][xp, y, xp, yp] + phis[s,f][xp, yp, x,yp] + phis[s,f][x,yp, x,y]
                fluxes_nn[x,y] = exp(im * flux)
                
                
                #### Next nearest neighbor triangles
                # up, right, down-left - yoni
                flux = phis[s,f][x, y, xp, yp] + phis[s,f][xp, yp, x, yp] + phis[s,f][x, yp, x,y]
                fluxes_nnn_1[x,y] = exp(im * flux)
                
                # down, left, up-right
                flux = phis[s,f][x, y, xm, ym] + phis[s,f][xm, ym, x, ym] + phis[s,f][x, ym, x,y]
                fluxes_nnn_2[x,y] = exp(im * flux)
                
                # left, up, down-right - yoni
                flux = phis[s,f][x, y, xm, yp] + phis[s,f][xm, yp, xm, y] + phis[s,f][xm, y, x, y]
                fluxes_nnn_3[x,y] = exp(im * flux)
                
                # right, down, up-left
                flux = phis[s,f][x, y, xp, ym] + phis[s,f][xp, ym, xp, y] + phis[s,f][xp, y, x,y]
                fluxes_nnn_4[x,y] = exp(im * flux)
                
                
                #### Next next nearest neighbor square
                # 2up, 2right, 2down, 2left
                flux = phis[s,f][x, y, xp2, y] + phis[s,f][xp2, y, xp2, yp2] + phis[s,f][xp2, yp2, x,yp2] + phis[s,f][x,yp2, x,y]
                fluxes_nnnn[x,y] = exp(im * flux)
                
                
                #### Other
                # up-right, down-right, 2left
                flux = phis[s,f][x, y, xp2, y] + phis[s,f][xp2, y, xp, yp] + phis[s,f][xp, yp, x,y]
                fluxes_mixed_1[x,y] = exp(im * flux)
                
                # up-left, up-right, 2down
                flux = phis[s,f][x, y, x, yp2] + phis[s,f][x, yp2, xm, yp] + phis[s,f][xm, yp, x,y]
                fluxes_mixed_2[x,y] = exp(im * flux)
            end
        end
        println(s," ",f)
        @test all( isapprox.(fluxes_nn, exp(im * B[s,f])) )

        if mc.p.Nhoppings != "none" && mc.p.NNhoppings != "none"
          @test all( isapprox.(fluxes_nnn_1, exp(im * B[s,f] / 2)) )
          @test all( isapprox.(fluxes_nnn_2, exp(im * B[s,f] / 2)) )
          @test all( isapprox.(fluxes_nnn_3, exp(im * B[s,f] / 2 )) )
          @test all( isapprox.(fluxes_nnn_4, exp(im * B[s,f] / 2)) )
          @test all( isapprox.(fluxes_nnnn, exp(im * 4 * B[s,f])) )
          @test all( isapprox.(fluxes_mixed_1, exp(im *  B[s,f])) )
          @test all( isapprox.(fluxes_mixed_2, exp(im *  B[s,f])) )
        end
      end
    end
end