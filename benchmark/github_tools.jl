using GitHub, Base64

GitHub.Content(::Nothing) = nothing

(@isdefined AUTH) || (const AUTH = GitHub.authenticate(ENV["GITHUB_AUTH"]))
(@isdefined REPO) || (const REPO = Repo("crstnbr/dqmc-benchmarks"))

function push(file::AbstractString;
            filename=file,
            message="push $filename"
            )
    content = read(file, String)
    d = Dict(
        "content" => base64encode(content), # maybe escape_string
        "message" => string(message, " (api)")
    )
    create_file(REPO, filename; auth=AUTH, params=d)
    nothing
end


function rm(filename::AbstractString;
            message="rm $filename"
            )
    d = Dict(
        "message" => string(message, " (api)"),
        "sha" => file(REPO, filename; auth=AUTH).sha
    )
    delete_file(REPO, filename; auth=AUTH, params=d)
    nothing
end

function rm(filenames::AbstractVector{T}) where T<:AbstractString
    for f in filenames
        @info "Removing $f"
        rm(f)
    end
    nothing
end

function download(filename)
    f = file(REPO, filename; auth=AUTH)
    c = replace(f.content, "\n"=>"")
    s = String(base64decode(c))
    write(filename, s)
    nothing
end


getfilelist() = GitHub.name.(directory(REPO, "."; auth=AUTH)[1])
function getresultfiles(; branch="")
    filter(x->occursin("result", x)&&occursin(branch, x)&&endswith(x, "json"), getfilelist())
end

clear_repo() = rm(filter(f->occursin("result", f), getfilelist()))