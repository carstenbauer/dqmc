using PkgBenchmark
result = PkgBenchmark.readresults(joinpath(@__DIR__, "result.json"))

using Markdown
# pretty-print to screen
# let io = IOBuffer()
#     PkgBenchmark.export_markdown(io, result)
#     seekstart(io)
#     md = Markdown.parse(io)
#     display(md)
# end

# pretty-print to file
@info "Pretty-printing markdown to file"
PkgBenchmark.export_markdown("result.md", result)