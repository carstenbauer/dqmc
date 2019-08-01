isfile("result.md") && Base.rm("result.md")
isfile("tune.json") && Base.rm("tune.json")
isfile("judgement.md") && Base.rm("judgement.md")

for f in readdir(@__DIR__)
    startswith(f, "result") || continue
    endswith(f, "json") || continue
    Base.rm(f)
end