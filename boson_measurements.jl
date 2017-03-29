function measure_op(s::Stack, p::Parameters, l::Lattice)
  mean_abs_op = mean(abs(p.hsfield))
  mean_op = vec(mean(p.hsfield,[2,3]))
  return (mean_abs_op, mean_op)
end

FFTW.set_num_threads(Sys.CPU_CORES)
# bosonic correlation function C(qy,qx,iw)
function measure_phi_correlations(s::Stack, p::Parameters, l::Lattice)
  phiFT = fft(reshape(p.hsfield,3,l.L,l.L,p.slices),[2,3,4])
  phiFT .*= conj(phiFT)
  n = Int(l.L/2+1)
  nt = Int(p.slices/2+1)
  return real(squeeze(sum(phiFT,1),1))[1:n,1:n,1:nt] # sum over op components
end

# This is the bosonic susceptibility X(qy-pi,qx-pi,iw)
#
# first element of C in w direction is 0 freq, last element of C in qx and qy direction is (-pi,-pi)
# (DSP.fftfreq(#elements, sampling rate = 1/a = 1 in our case))
# just reverse qx and qy dimensions to get q-Q dependence
function measure_chi(s::Stack, p::Parameters, l::Lattice)
  mapslices(x->rotr90(rotr90(x)),measure_phi_correlations(s,p,l),[1,2])
end

# X(Q,0) = X(pi,pi,0) = C(0,0)
function measure_chi_static(s::Stack, p::Parameters, l::Lattice)
  measure_phi_correlations(s,p,l)[1,1,1] # 0 component = idx 1 element in Julia
end

# This is slightly FASTER than fft variant above
function measure_chi_static_direct(s::Stack, p::Parameters, l::Lattice)
  chi = 0.0
  for i in 1:l.sites
    for j in 1:l.sites
      for t1 in 1:p.slices
        for t2 in 1:p.slices
          chi += dot(p.hsfield[:,i,t1], p.hsfield[:,j,t2])
        end
      end
    end
  end
  return chi
end