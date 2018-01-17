function measure_op(conf::AbstractArray{Float64, 3})
  mean_abs_op = mean(abs.(conf))
  mean_op = vec(mean(conf,[2,3]))
  return mean_abs_op, mean_op
end

FFTW.set_num_threads(Sys.CPU_CORES)
# bosonic correlation function C(qy,qx,iw), not normalized
function measure_phi_correlations(conf::AbstractArray{Float64, 3})
  opdim, sites, slices = size(conf)
  L = Int(sqrt(sites))
  phiFT = rfft(reshape(conf,opdim,L,L,slices),[2,3,4])
  phiFT .*= conj(phiFT)
  n = floor(Int, L/2+1)
  nt = floor(Int, slices/2+1)
  return real(sum(phiFT,1))[1,:,1:n,1:nt] # sum over op components
end

import DSP.rfftfreq
# returns qys, qxs and ωs corresponding to output of measure_phi_correlations etc.
function get_momenta_and_frequencies(L::Int, M::Int, Δτ::Float64=0.1)
  # https://juliadsp.github.io/DSP.jl/latest/util.html#DSP.Util.rfftfreq
  qs = 2*pi*rfftfreq(L) # 2*pi because rfftfreq returns linear momenta
  ωs = 2*pi*rfftfreq(M, 1/Δτ)
  return qs, qs, ωs
end
get_momenta_and_frequencies(C::AbstractArray{Float64, 3}, Δτ::Float64=0.1) = get_momenta_and_frequencies(size(C,2,3)..., Δτ)

# χ(qy,qx,iw) = C(qy, qx, iw) * Δτ/(N*M)
function measure_chi_dynamic(conf::AbstractArray{Float64, 3}; Δτ::Float64=0.1)
  const N = size(conf, 2)
  const M = size(conf, 3)
  Δτ/(N*M) * measure_phi_correlations(conf)
end

# This is the rotated bosonic susceptibility with momenta relative to Q,
# i.e. χ_rotated(qy,qx,iw) = C(qy-pi, qx-pi, iw) * Δτ/(N*M)
#
# first element of C in w direction is 0 freq, last element of C in qx and qy direction is (-pi,-pi)
# (DSP.fftfreq(#elements, sampling rate = 1/a = 1 in our case))
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
  m = mean(conf,[2,3]) # average over sites and timeslices
  m2 = dot(m,m)
  return (m2, m2*m2)
end