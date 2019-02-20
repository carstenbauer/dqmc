# Pure tests of functions defined in linalg.jl
using JLD, BenchmarkTools

"""
Build companion matrix from a vector.
"""
function compan(c::AbstractVector{T}) where T
    N = length(c)
    A = diagm(-1 => ones(T, N-1))
    @inbounds for i in 1:N
        A[1,i] = - c[i]/c[1]
    end
    A
end

# generate a nasty ill-conditioned matrix
N = 27
(@isdefined IC) || (const IC = compan(reverse( vcat(1, 1 ./ cumprod(1:N, dims=1)) )))


# B-chain in the region where stabilization is necessary
B_SVD_U = load("data/linalg.jld", "B_SVD_U")
B_SVD_D = load("data/linalg.jld", "B_SVD_D")
B_SVD_Vt = load("data/linalg.jld", "B_SVD_Vt")

B_QR_U = load("data/linalg.jld", "B_QR_U")
B_QR_D = load("data/linalg.jld", "B_QR_D")
B_QR_T = load("data/linalg.jld", "B_QR_T")




@testset "Linear Algebra" begin
    
    @testset "Decompositions" begin

        @testset "QR aka UDT" begin
            Random.seed!(1234)
            x = rand(10,10)

            u,d,t = decompose_udt(x)

            @test isapprox(u * u', I) # unitarity of u
            @test sort!(count.(iszero,eachcol(t))) == 0:9 # triangularity of t (istriu(t) == false due to pivoting)
            @test d == [2.0718140718296794, 1.3713719555859616, 1.1527980473310226, 1.1359672199189663,
             0.9230156804883671, 0.7796143878032683, 0.5299931859601968, 0.43829746451880564,
              0.36921251585078685, 0.13351187833543765]
            @test isapprox(u * Diagonal(d) * t, x)


            D = zero(d);
            u, t = decompose_udt!(copy(x), D)

            @test isapprox(u * u', I) # unitarity of u
            @test sort!(count.(iszero,eachcol(t))) == 0:9 # triangularity of t (istriu(t) == false due to pivoting)
            @test D == [2.0718140718296794, 1.3713719555859616, 1.1527980473310226, 1.1359672199189663,
             0.9230156804883671, 0.7796143878032683, 0.5299931859601968, 0.43829746451880564,
              0.36921251585078685, 0.13351187833543765]
            @test isapprox(u * Diagonal(D) * t, x)
        end


        @testset "SVD aka UDV" begin
            Random.seed!(1234)
            x = rand(10,10)

            u,d,vt = decompose_udv(x)

            @test isapprox(u * u', I) # unitarity of u
            @test isapprox(vt * vt', I) # unitarity of v
            @test isapprox(u * Diagonal(d) * vt, x)
            @test d == [4.69085124499646, 1.3700826637207126, 1.2218064335214107, 1.069747771166964,
             0.8846933503887872, 0.7163644709197314, 0.5338397904461203, 0.4033960707621825,
              0.2653967719523671, 0.10076327879773647]

            u2, d2, vt2 = decompose_udv!(x)

            @test isapprox(u, u2)
            @test isapprox(d, d2)
            @test isapprox(vt, vt2)
        end
    end



    @testset "Dense-Sparse in-place speed" begin
        # https://discourse.julialang.org/t/asymmetric-speed-of-in-place-sparse-dense-matrix-product/10256/5
        A = sprand(1000,1000,0.01);
        B = rand(1000,1000);
        C = similar(B);

        # check that the order doesn't matter much
        sparse_dense = @belapsed mul!($C,$A,$B);
        dense_sparse = @belapsed mul!($C,$A,$B);
        @test abs(sparse_dense - dense_sparse)/(sparse_dense+dense_sparse) < 0.7
        # let's hope that this thresh works on other machines
    end


    @testset "Safe multiply UDT and UDV" begin
        # UDT - calculating square of ill-conditioned slice matrix
        u = B_QR_U; d = B_QR_D; t = B_QR_T;
        # naive = (u * Diagonal(d) * t)^2
        # u_naive, d_naive, t_naive = decompose_udt(naive)
        U,D,T = multiply_safely(u,d,t,u,d,t)
        @test isapprox(maximum(D), 5.8846316709257896e16)
        @test isapprox(minimum(D), 1.9123571535539083e-24)
    end


    @testset "UDT and UDV inversion business" begin

    end
end