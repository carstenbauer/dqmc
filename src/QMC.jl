# fake a package
module QMC

__precompile__(false)

include("dqmc_framework.jl")

function mc_from_inxml(inxml::AbstractString)
    Random.seed!(1234)
    if SUPPRESS_DQMC_INIT_OUTPUT
        old_stdout = stdout
        (rp, wp) = redirect_stdout()
    end

    @assert isfile(inxml)
    p = Params()
    p.output_file = "$(inxml[1:end-7]).out.h5.running"
    xml2parameters!(p, inxml)
    mc = DQMC(p)
    init!(mc)

    if SUPPRESS_DQMC_INIT_OUTPUT
        redirect_stdout(old_stdout)
        close(wp); close(rp);
    end
    return mc
end

function mc_from_inxml(inxml::AbstractString, initconf)
    mc = mc_from_inxml(inxml)
    mc.p.hsfield .= initconf
    return mc
end


if !haskey(ENV, "JULIA_DQMC")
    ENV["JULIA_DQMC"] = dirname(dirname(@__FILE__)) # parent directory
end


(@isdefined SUPPRESS_DQMC_INIT_OUTPUT) || (const SUPPRESS_DQMC_INIT_OUTPUT = false)

# exportall:
for n in names(@__MODULE__, all=true)
    if Base.isidentifier(n) && n ∉ (Symbol(@__MODULE__), :eval)
        @eval export $n
    end
end

end
