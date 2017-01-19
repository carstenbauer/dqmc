function setGLOBAL_RNG(rng::MersenneTwister)
  Base.Random.GLOBAL_RNG.idx = rng.idx
  Base.Random.GLOBAL_RNG.state = rng.state
  Base.Random.GLOBAL_RNG.vals = rng.vals
  Base.Random.GLOBAL_RNG.seed = rng.seed
  nothing
end
