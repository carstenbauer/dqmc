using DataFrames, CSV
# using Pkg
# Pkg.activate(ENV["JULIA_DQMC"])
push!(LOAD_PATH, ENV["JULIA_DQMC"])

include(joinpath(ENV["JULIA_DQMC"], "src/dqmc_framework.jl"))

SAFE_MULT = 10

function estimate_memory_usage_tdgfs()
    df = DataFrame(L=Int[], B=Int[], mem=Float64[])

    for L in [4,8,10,12,14]
        for beta in [5,10,20,40]
            mem = estimate_memory_usage_tdgfs(L, beta; safe_mult=SAFE_MULT)
            push!(df,[L, beta, mem])
        end
    end

    return df
end

df = estimate_memory_usage_tdgfs()

CSV.write("tdgfs_memory_safe_mult_$(SAFE_MULT).csv", df)









using PyPlot, Polynomials
plt.style.use("publication_tex")
# plt.style.use("default")

function plot_memusage(df)
    g = groupby(df, :L);

    for (i, gl) in enumerate(g)
        x = gl[:B]
        y = gl[:mem] ./ 1024 # MB -> GB
        p = polyfit(x, y, 1)
        m = round(p.a[2], digits=3)
        b = round(p.a[1], digits=3)
        plot(x, y, "o-", color="C$i", label="L = $(gl[:L][1]) ($(m)x + $b)")
    end
    handles, labels = gca().get_legend_handles_labels()
    legend(reverse(handles), reverse(labels), frameon=false, prop=Dict("size" => 15))
    xlim([3,42])
    # ylim([-0.7, 11])
    xlabel(L"inverse temperature $\beta$")
    ylabel("tdgfs memory usage in GB")
    tight_layout()
    savefig("tdgfs_memory_safe_mult_$(SAFE_MULT).pdf")
    nothing
end

plot_memusage(df)
