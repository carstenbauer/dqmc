# -------------------------------------------------------
#         Measurements
# -------------------------------------------------------

# χ(qy,qx,iw) = C(qy, qx, iw) * Δτ/(N*M)
function measure_chi_dynamic(conf::AbstractArray{Float64, 3}; Δτ::Float64=0.1)
  N = size(conf, 2)
  M = size(conf, 3)
  Δτ/(N*M) * measure_phi_correlations(conf)
end

# don't sum over op components
function measure_chi_dynamic_components(conf::AbstractArray{Float64, 3}; Δτ::Float64=0.1)
  N = size(conf, 2)
  M = size(conf, 3)
  Δτ/(N*M) * measure_phi_correlations_components(conf)
end

# This is the rotated bosonic susceptibility with momenta relative to Q,
# i.e. χ_rotated(qy,qx,iw) = C(qy-pi, qx-pi, iw) * Δτ/(N*M)
#
# first element of C in w direction is 0 freq, last element of C in qx and qy direction is (-pi,-pi)
# (fftfreq(#elements, sampling rate = 1/a = 1 in our case))
# just reverse qx and qy dimensions to get q-Q dependence
function measure_rotated_chi_dynamic(conf::AbstractArray{Float64, 3}; Δτ::Float64=0.1)
  mapslices(x->rotr90(rotr90(x)), measure_chi_dynamic(conf; Δτ=Δτ), [1,2])
end

# χ_rotated(Q,0) = χ_rotated(pi,pi,0) = χ(0,0) = C(0,0) * Δτ/(N*M)
function measure_chi_static(conf::AbstractArray{Float64, 3}; Δτ::Float64=0.1)
  measure_chi_dynamic(conf; Δτ=Δτ)[1,1,1] # mean component == 0 component == idx 1 element in Julia
end

# Binder factors (m^2, m^4)
function measure_binder_factors(conf::AbstractArray{Float64, 3})
  m = mean(conf,dims=(2,3)) # average over sites and timeslices
  m2 = dot(m,m)
  return (m2, m2*m2)
end

function measure_op(conf::AbstractArray{Float64, 3})
  mean_abs_op = mean(abs.(conf))
  mean_op = vec(mean(conf,dims=(2,3)))
  return mean_abs_op, mean_op
end

# bosonic correlation function C(qy,qx,iw), not normalized
function measure_phi_correlations(conf::AbstractArray{Float64, 3})
  opdim, sites, slices = size(conf)
  L = Int(sqrt(sites))
  phiFT = rfft(reshape(conf,opdim,L,L,slices),[2,3,4])
  phiFT .*= conj(phiFT)
  n = floor(Int, L/2+1)
  nt = floor(Int, slices/2+1)
  return real(sum(phiFT,dims=1))[1,:,1:n,1:nt] # sum over op components
end

# measure correlations for every op component (don't sum over them)
function measure_phi_correlations_components(conf::AbstractArray{Float64, 3})
  opdim, sites, slices = size(conf)
  L = Int(sqrt(sites))
  phiFT = rfft(reshape(conf,opdim,L,L,slices),[2,3,4])
  phiFT .*= conj(phiFT)
  n = floor(Int, L/2+1)
  nt = floor(Int, slices/2+1)
  return real(phiFT)[:,:,1:n,1:nt] # do not yet sum over op components
end



# -------------------------------------------------------
#         Postprocessing/Analysis
# -------------------------------------------------------

"""
Returns qys, qxs and ωs corresponding to output of measure_phi_correlations etc.
"""
function get_momenta_and_frequencies(L::Int, M::Int, Δτ::Float64=0.1)
  qs = 2*pi*rfftfreq(L) # 2*pi because rfftfreq returns linear momenta
  ωs = 2*pi*rfftfreq(M, 1/Δτ)
  return qs, qs, ωs
end
get_momenta_and_frequencies(C::AbstractArray{Float64, 3}, Δτ::Float64=0.1) = get_momenta_and_frequencies(Int(sqrt(size(C,2))), size(C,3), Δτ)
get_momenta_and_frequencies(mc::AbstractDQMC) = get_momenta_and_frequencies(mc.l.L, mc.p.slices, mc.p.delta_tau)












# ------------------------------------------------------------------
#         "Macroscopic methods" - measure MC time series
# ------------------------------------------------------------------
function measure_chi_dynamics(confs::AbstractArray{Float64, 4})
    num_confs = size(confs,4)
    N = size(confs, 2)
    M = size(confs, 3)

    chi_dyn_symm = Observable(Array{Float64, 3}, "chi_dyn_symm"; alloc=num_confs)
    chi_dyn = Observable(Array{Float64, 3}, "chi_dyn"; alloc=num_confs)
    @inbounds @views for i in 1:num_confs
        chi = measure_chi_dynamic(confs[:,:,:,i])
        push!(chi_dyn, chi)
        chi = (permutedims(chi, [2,1,3]) + chi)/2 # C4 is basically flipping qx and qy (which only go from 0 to pi since we perform a real fft.)
        push!(chi_dyn_symm, chi)
    end

    return chi_dyn, chi_dyn_symm
end

function measure_binder(confs::AbstractArray{Float64, 4})
    num_confs = size(confs,4)
    N = size(confs, 2)
    M = size(confs, 3)

    m2s = Vector{Float64}(undef, num_confs)
    m4s = Vector{Float64}(undef, num_confs)
    @inbounds @views for i in 1:num_confs
        m = mean(confs[:,:,:,i],dims=(2,3))
        m2s[i] = dot(m, m)
        m4s[i] = m2s[i]*m2s[i]
    end

    m2ev2 = mean(m2s)^2
    m4ev = mean(m4s)

    binder = Observable(Float64, "binder")
    push!(binder, m4ev/m2ev2)

    return binder
end
