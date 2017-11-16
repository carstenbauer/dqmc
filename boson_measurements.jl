function measure_op(conf::Array{Float64, 3})
  mean_abs_op = mean(abs.(conf))
  mean_op = vec(mean(conf,[2,3]))
  return mean_abs_op, mean_op
end

FFTW.set_num_threads(Sys.CPU_CORES)
# bosonic correlation function C(qy,qx,iw)
function measure_phi_correlations(conf::Array{Float64, 3})
  opdim, sites, slices = size(conf)
  L = Int(sqrt(sites))
  phiFT = fft(reshape(conf,opdim,L,L,slices),[2,3,4])
  phiFT .*= conj(phiFT)
  n = Int(L/2+1)
  nt = Int(slices/2+1)
  return real(sum(phiFT,1))[1,1:n,1:n,1:nt] # sum over op components
end

# This is the bosonic susceptibility X(qy-pi,qx-pi,iw)
#
# first element of C in w direction is 0 freq, last element of C in qx and qy direction is (-pi,-pi)
# (DSP.fftfreq(#elements, sampling rate = 1/a = 1 in our case))
# just reverse qx and qy dimensions to get q-Q dependence
function measure_chi(conf::Array{Float64, 3})
  mapslices(x->rotr90(rotr90(x)),measure_phi_correlations(conf),[1,2])
end

# X(Q,0) = X(pi,pi,0) = C(0,0)
function measure_chi_static(conf::Array{Float64, 3})
  measure_phi_correlations(conf)[1,1,1] # 0 component = idx 1 element in Julia
end

# Binder factors (m^2, m^4)
function measure_binder_factors(conf::Array{Float64, 3})
  m = mean(conf,[2,3]) # average over sites and timeslices
  m2 = dot(m,m)
  return (m2, m2*m2)
end