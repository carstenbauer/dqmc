@testset "Dense-Sparse in-place speed bug" begin
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