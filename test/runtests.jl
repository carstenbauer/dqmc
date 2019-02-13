using Test
include("../dqmc_framework.jl") # to be replaced by using DQMC or similar

function mc_from_inxml(inxml::AbstractString)
    @assert isfile(inxml)
    p = Params()
    p.output_file = "$(inxml[1:end-7]).out.h5.running"
    xml2parameters!(p, inxml)
    mc = DQMC(p)
    init!(mc)
    mc
end

function mc_from_inxml(inxml::AbstractString, initconf)
    @assert isfile(inxml)
    p = Params()
    p.output_file = "$(inxml[1:end-7]).out.h5.running"
    xml2parameters!(p, inxml)
    mc = DQMC(p)
    init!(mc)

    mc.p.hsfield .= initconf
    mc
end

# tests
include("O3.jl")

# no interactions
include("freefermions.jl")

nothing