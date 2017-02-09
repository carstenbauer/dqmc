using HDF5


new_op = h5read("DetRatioCheck.h5", "new_op")
site = h5read("DetRatioCheck.h5", "site")
slice = h5read("DetRatioCheck.h5", "slice")
p.hsfield = h5read("DetRatioCheck.h5", "hsfield")
old_op = p.hsfield[:,site,slice]

h5write("DetRatioCheck.h5", "new_op", new_op)
h5write("DetRatioCheck.h5", "site", site)
h5write("DetRatioCheck.h5", "N", l.sites)
h5write("DetRatioCheck.h5", "slice", slice)
h5write("DetRatioCheck.h5", "delta_tau", p.delta_tau)
h5write("DetRatioCheck.h5", "lambda", p.lambda)
h5write("DetRatioCheck.h5", "hsfield", p.hsfield)

V = interaction_matrix_exp(p,l,slice)

h5write("DetRatioCheck.h5", "V_real", real(V))
h5write("DetRatioCheck.h5", "V_imag", imag(V))

hsfield = copy(p.hsfield)
newhsfield = copy(p.hsfield)
newhsfield[:,site,slice] = new_op[:]
p.hsfield[:] = newhsfield[:]
newV = interaction_matrix_exp(p,l,slice)
p.hsfield[:] = hsfield[:]

h5write("DetRatioCheck.h5", "newV_real", real(newV))
h5write("DetRatioCheck.h5", "newV_imag", imag(newV))

s.current_slice = slice
detratio = calculate_detratio(s,p,l,site,new_op)
deltai = copy(s.delta_i)
M = copy(s.M)

h5write("DetRatioCheck.h5", "detratio_real", real(detratio))
h5write("DetRatioCheck.h5", "detratio_imag", imag(detratio))
h5write("DetRatioCheck.h5", "deltai_real", real(deltai))
h5write("DetRatioCheck.h5", "deltai_imag", imag(deltai))
h5write("DetRatioCheck.h5", "M_real", real(M))
h5write("DetRatioCheck.h5", "M_imag", imag(M))
