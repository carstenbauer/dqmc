using Dates, Git

include("github_tools.jl")


@info "Uploading results to crstnbr/dqmc-benchmarks."
if isfile("result.json")
    dt = string(now())
    dt = replace(dt, ":"=>"--")
    branch = Git.branch()
    push("result.json"; filename="result_$(branch)_$(dt).json")
    if isfile("result.md")
        push("result.md"; filename="result_$(branch)_$(dt).md")
    else
        @warn "uploadresult.jl: No result.md. Pushed only result.json."
    end
else
    @warn "uploadresult.jl: Couldn't find result.json."
end