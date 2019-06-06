using Pkg
Pkg.activate(ENV["JULIA_DQMC"])

include(joinpath(ENV["JULIA_DQMC"], "src/dqmc_framework.jl"))

input = "input.in.xml" # choose DQMC input file

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



using PyPlot, Polynomials
plt.style.use("publication_tex")
# plt.style.use("default")

function plot_memusage(df)
    g = groupby(df, :L);

    for (i, gl) in enumerate(g)
        x = gl[:B]
        y = gl[:mem]
        p = polyfit(x, y, 1)
        m = round(p.a[2], digits=1)
        b = round(p.a[1], digits=1)
        plot(x, y, "o-", color="C$i", label="L = $(gl[:L][1]) ($(m)x + $b)")
        # plot(1:40, p.(1:40), color="k")
    end
    handles, labels = gca().get_legend_handles_labels()
    legend(reverse(handles), reverse(labels), frameon=false, prop=Dict("size" => 15))
    xlim([3,42])
    ylim([-100, 1800])
    xlabel(L"inverse temperature $\beta$")
    ylabel("static memory usage in MB")
    tight_layout()
    savefig("mc_memory.pdf")
    nothing
end

plot_memusage(df)