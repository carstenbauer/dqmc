###############################################################
#                Combined Mean and Variance
###############################################################
"""
    combined_mean_and_var(xs...) -> meanc, varc
Calculates the combined mean and variance of the concatenated sample `vcat(xs...)`.
"""
function combined_mean_and_var(xstuple::AbstractVector{T}...) where T
    xs = collect(xstuple)
    ns = length.(xs)
    μs = mean.(xs)
    vs = var.(xs)
    return combined_mean_and_var(ns, μs, vs)
end

export combined_mean_and_var




"""
    combined_mean_and_var(ns, μs, vs) -> meanc, varc
Given N samples characterized by their lengths `ns`,
their means `μs`, and their variances `vs`,
calculates the combined (or pooled) mean and variance of the
overall (concatenated) sample.
"""
function combined_mean_and_var(ns::AbstractVector{<:Integer},
                               μs::AbstractVector{<:Number},
                               vs::AbstractVector{<:Number})
    nsum = sum(ns)
    meanc = dot(ns, μs) / nsum
    varc = sum((ns .- 1) .* vs + ns .* abs2.(μs .- meanc)) / (nsum - 1)
    return meanc, varc
end


function combined_mean_and_var(ns::AbstractVector{<:Integer},
                               μs::AbstractVector{<:AbstractArray{<:Number}},
                               vs::AbstractVector{<:AbstractArray{<:Number}})
    meanc = zero(μs[1])
    varc = zero(vs[1])

    nsum = sum(ns)
    N = length(μs) # number of samples
    for i in eachindex(meanc)

        for k in 1:N
            meanc[i] += ns[k] * μs[k][i]
        end
        meanc[i] = meanc[i] / nsum


        for k in 1:N
            varc[i] += (ns[k] - 1) * vs[k][i] + ns[k] * abs2(μs[k][i] - meanc[i])
        end
        varc[i] = varc[i] / (nsum - 1)
    end

    return meanc, varc
end




###############################################################
#       Combined Mean and Variance ("Hand-written")
###############################################################
"""
    combined_mean_and_var(x1, x2) -> meanc, varc
Given two samples `x1`,`x2` calculates the mean and variance of the
concatenated sample.
"""
function combined_mean_and_var(x1::AbstractVector{<:Number}, x2::AbstractVector{<:Number})
    n1, n2 = length(x1), length(x2)
    μ1, μ2 = mean(x1), mean(x2)
    v1, v2 = var(x1), var(x2)
    return combined_mean_and_var(n1, μ1, v1, n2, μ2, v2)
end


"""
    combined_mean_and_var(n1, μ1, v1, n2, μ2, v2) -> meanc, varc
Given two samples characterized by their lengths `n1`, `n2`, 
their means `μ1`, `μ2`, and their variances `v1`, `v2`,
calculates the combined (or pooled) mean and variance of the
concatenated sample.
"""
function combined_mean_and_var(n1::Integer, μ1::Number, v1::Number,
                               n2::Integer, μ2::Number, v2::Number)
    meanc = (n1 * μ1 + n2 * μ2) / (n1 + n2)

    # Based on https://www.emathzone.com/tutorials/basic-statistics/combined-variance.html,
    # including Robert Matheson's comment and adding abs for complex number support.
    varc = ((n1-1)*v1 + (n2-1)*v2 + n1*abs2(μ1 - meanc) + n2*abs2(μ2 - meanc)) /
                                (n1 + n2 - 1)
    return meanc, varc
end


# Explicit version for three samples (only for testing)
function combined_mean_and_var_three(x1, x2, x3)
    n1, n2, n3 = length(x1), length(x2), length(x3)
    μ1, μ2, μ3 = mean(x1), mean(x2), mean(x3)
    v1, v2, v3 = var(x1), var(x2), var(x3)

    meanc12, varc12 = combined_mean_and_var(n1, μ1, v1, n2, μ2, v2)
    n12 = n1 + n2
    return combined_mean_and_var(n12, meanc12, varc12, n3, μ3, v3)
end










"""
    combined_mean_and_var(ors::Vector{<:ObservableResult}) -> meanc, stderrc
Calculates the combined mean and standard error of the concatenated observable results.
"""
function combined_mean_and_error(ors::Vector{<:ObservableResult})
    ns = [r.count for r in ors]
    μs = [r.mean for r in ors]
    vs = [r.error^2 for r in ors]

    meanc, varc = combined_mean_and_var(ns, μs, vs)
    return meanc, sqrt.(varc)
end