using PkgBenchmark, Random


include("cleanup.jl")

download("tune.json")

@info "Running benchmarks"
benchmarkpkg(
    dirname(@__DIR__),
    BenchmarkConfig(
        env = Dict(
            "JULIA_NUM_THREADS" => "1",
            "OMP_NUM_THREADS" => "1",
        ),
    ),
    resultfile = joinpath(@__DIR__, "result.json"),
)

include("pprintresult.jl")
include("github_tools.jl")
include("judgebenchmark.jl")
## include("uploadresult.jl")