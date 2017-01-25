function setGLOBAL_RNG(rng::MersenneTwister)
  Base.Random.GLOBAL_RNG.idx = rng.idx
  Base.Random.GLOBAL_RNG.state = rng.state
  Base.Random.GLOBAL_RNG.vals = rng.vals
  Base.Random.GLOBAL_RNG.seed = rng.seed
  nothing
end


function sparsity(A::AbstractArray)
  length(findin(A,0))/length(A)
end


function swapRows!(X, i, j)
    for k = 1:size(X,2)
        X[i,k], X[j,k] = X[j,k], X[i,k]
    end
end
