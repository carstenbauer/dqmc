using PkgBenchmark, Dates

function parse_datetime(resultfile)
    dtstring = split(split(resultfile, "_")[3], ".json")[1]
    dt = DateTime(replace(dtstring, "--"=>":"))
end

function sortdesc!(resultfiles)
    dts = parse_datetime.(resultfiles)
    resultfiles .= resultfiles[sortperm(dts; rev=true)]
end

function judge_results(target_file, baseline_file)
    j = judge(readresults(target_file), readresults(baseline_file))
    PkgBenchmark.export_markdown("judgement.md", j)
    j
end

resultfiles = sortdesc!(getresultfiles(branch="benchmark"))
if length(resultfiles) > 0
    @info "Downloading latest baseline"
    latest = resultfiles[1]
    download(latest)

    @info "Exporting judgement as markdown"
    judge_results("result.json", latest)
else
    @warn "No result files found in repo."
end