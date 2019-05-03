using Pkg
Pkg.activate(ENV["JULIA_DQMC"])

include(joinpath(ENV["JULIA_DQMC"], "src/dqmc_framework.jl"))

input = "dqmc.in.xml" # choose DQMC input file

using Suppressor, DataFrames, CSV


function measure_mc_memory_usage()
    df = DataFrame(L=Int[], B=Int[], mem=Float64[])
    
    @suppress for L in [4,8,10,12,14]
        for beta in [5,10,20,40]
            p = Params()
            p.output_file = "live.out.h5.running"
            xml2parameters!(p, input)
            p.lattice_file = ""
            p.L = L
            p.beta = beta
            p.slices = beta * 10
            deduce_remaining_parameters(p)
            mc = DQMC(p)
            init!(mc)

            push!(df, [L, beta, memory_usage(mc)])
        end
    end 
    
    return df
end

df = measure_mc_memory_usage()

CSV.write("mc_memory.csv", df)