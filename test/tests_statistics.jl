@testset "Combined Mean and Variance" begin

    @testset "Two samples" begin
        test_two = (x1,x2) -> begin
            xc = vcat(x1,x2)
            meanc, varc = combined_mean_and_var(x1,x2)
            @test abs(mean(xc) - meanc) < 1e-12
            @test abs(var(xc) - varc) < 1e-12
        end

        x1 = rand(30_000)
        x2 = rand(20_000)
        test_two(x1,x2)

        x1 = rand(ComplexF64, 30_000)
        x2 = rand(ComplexF64, 20_000)
        test_two(x1,x2)
    end


    @testset "Three samples" begin
        test_three = (x1,x2,x3) -> begin
            xc = vcat(x1,x2,x3)
            meanc, varc = combined_mean_and_var(x1,x2,x3)
            @test abs(mean(xc) - meanc) < 1e-12
            @test abs(var(xc) - varc) < 1e-12

            meanc_three, varc_three = combined_mean_and_var_three(x1,x2,x3)
            @test isapprox(meanc, meanc_three)
            @test isapprox(varc, varc_three)
        end

        x1 = rand(30_000)
        x2 = rand(20_000)
        x3 = rand(40_000)
        test_three(x1, x2, x3)

        x1 = rand(ComplexF64, 30_000)
        x2 = rand(ComplexF64, 20_000)
        x3 = rand(ComplexF64, 40_000)
        test_three(x1, x2, x3)
    end


    @testset "N samples" begin
        test_N = (xs...) -> begin
            xc = vcat(xs...)
            meanc, varc = combined_mean_and_var(xs...)
            @test abs(mean(xc) - meanc) < 1e-12
            @test abs(var(xc) - varc) < 1e-12
        end

        lengths = [30_000, 20_000, 40_000]
        N = 5

        xs = [rand(Float64, rand(lengths)) for _ in 1:N]
        test_N(xs...)

        xs = [rand(ComplexF64, rand(lengths)) for _ in 1:N]
        test_N(xs...)
    end


    @testset "N samples (ns, μs, vs)" begin
        test_N_moments = (ns, μs, vs, mean_exact, var_exact) -> begin
            meanc, varc = combined_mean_and_var(ns, μs, vs)
            @test abs(mean_exact - meanc) < 1e-12
            @test abs(var_exact - varc) < 1e-12
        end

        lengths = [30_000, 20_000, 40_000]
        N = 5

        xs = [rand(Float64, rand(lengths)) for _ in 1:N]
        xc = vcat(xs...)
        test_N_moments(length.(xs), mean.(xs), var.(xs), mean(xc), var(xc))

        xs = [rand(ComplexF64, rand(lengths)) for _ in 1:N]
        xc = vcat(xs...)
        test_N_moments(length.(xs), mean.(xs), var.(xs), mean(xc), var(xc))
    end
end