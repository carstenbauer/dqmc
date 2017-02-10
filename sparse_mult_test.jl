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


# check wether for a given matrix size and # of nz-element
# sparse * dense is faster/slower than dense * dense
function test_sparse_dense_vs_dense_dense(msize::Int=100, nzelements::Int=8, samplesize::Int=50)
  x = spzeros(msize,msize);
  # the hopping matrices Tx_As are NxN with exactly 8 (= four site hopping * back and forth) non-zero entries
  x[rand(1:length(x),nzelements)] = rand(nzelements);
  # x = sprand(msize,msize,.1);
  fx = full(x);
  y = rand(msize,msize);
  # xa = Base.SparseArrays.CHOLMOD.Sparse(x);

  times_normal = zeros(samplesize)
  times_sparse = zeros(samplesize)
  # times_sparse_alt = zeros(samplesize)
  times_sparse_sparse = zeros(samplesize)
  times_normal_normal = zeros(samplesize)
  for k in 1:samplesize
    tic()
    fx * y
    times_normal[k] = toq()

    tic()
    x * y
    times_sparse[k] = toq()

    # tic()
    # xa * y
    # times_sparse_alt[k] = toq()

    tic()
    x * x
    times_sparse_sparse[k] = toq()

    tic()
    fx * fx
    times_normal_normal[k] = toq()
  end

  t_normal = mean(times_normal)
  t_sparse = mean(times_sparse)

  @printf("normal mean: %.1es\n", t_normal)
  @printf("sparse mean: %.1es\n", t_sparse)
  if t_sparse < t_normal
    @printf("sparse speedup factor: %.1f\n", abs(t_normal/t_sparse))
  else
    @printf("sparse slowdown(!) factor: %.1f\n", abs(t_sparse/t_normal))
  end
  # println("")
  # println("sparse alt mean: ", mean(times_sparse_alt))
  # println("sparse sparse mean: ", mean(times_sparse_sparse))
  # println("normal normal mean: ", mean(times_normal_normal))

  # return times_normal, times_sparse, times_sparse_sparse, times_normal_normal
  return sparsity(x)
end
# Conclusion: sparsity is not a good enough criterium for if we will gain something
#             or not.
