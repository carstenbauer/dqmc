spmats = Vector{SparseMatrixCSC{Float64,Int64}}(100)
mats = Vector{Matrix{Float64}}(100)
m = rand(500,500)
for k in 1:100
  spmats[k] = sprand(500,500,.2)
  mats[k] = full(spmats[k])
end


function test_normal(x::Vector{Matrix{Float64}}, y::Matrix{Float64})
  for m in x
    m * y
  end
end


function test_sparse(x::Vector{SparseMatrixCSC{Float64,Int64}}, y::Matrix{Float64})
  for m in x
    m * y
  end
end

@time test_normal(mats, m);
@time test_normal(mats, m);
@time test_normal(mats, m);

@time test_sparse(spmats, m);
@time test_sparse(spmats, m);
@time test_sparse(spmats, m);

println("done")


function test_simple_sparse_vs_normal()
  x = sprand(1000,1000,.1);
  fx = full(x);
  y = rand(1000,1000);
  xa = Base.SparseArrays.CHOLMOD.Sparse(x);

  times_normal = zeros(10)
  times_sparse = zeros(10)
  times_sparse_alt = zeros(10)
  times_sparse_sparse = zeros(10)
  times_normal_normal = zeros(10)
  for k in 1:10
    tic()
    fx * y
    times_normal[k] = toq()

    tic()
    x * y
    times_sparse[k] = toq()

    tic()
    xa * y
    times_sparse_alt[k] = toq()

    tic()
    x * x
    times_sparse_sparse[k] = toq()

    tic()
    fx * fx
    times_normal_normal[k] = toq()
  end

  println("normal mean: ", mean(times_normal))
  println("sparse mean: ", mean(times_sparse))
  println("sparse alt mean: ", mean(times_sparse_alt))
  println("sparse sparse mean: ", mean(times_sparse_sparse))
  println("normal normal mean: ", mean(times_normal_normal))

  # return times_normal, times_sparse, times_sparse_alt, times_sparse_sparse, times_normal_normal
end
# Understand: Why does sparse product take longer????!!
