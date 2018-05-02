using HDF5


# new_op = h5read("tests/DetRatioCheck.h5", "new_op")
# site = h5read("tests/DetRatioCheck.h5", "site")
# slice = h5read("tests/DetRatioCheck.h5", "slice")
# p.hsfield = h5read("tests/DetRatioCheck.h5", "hsfield")
new_op = rand(3)
site = rand(1:l.sites)
slice = rand(1:p.slices)
old_op = p.hsfield[:,site,slice]

h5write("tests/DetRatioCheck.h5", "new_op", new_op)
h5write("tests/DetRatioCheck.h5", "site", site)
h5write("tests/DetRatioCheck.h5", "N", l.sites)
h5write("tests/DetRatioCheck.h5", "slice", slice)
h5write("tests/DetRatioCheck.h5", "delta_tau", p.delta_tau)
h5write("tests/DetRatioCheck.h5", "lambda", p.lambda)
h5write("tests/DetRatioCheck.h5", "hsfield", p.hsfield)

V = interaction_matrix_exp(p,l,slice)
Vminus = interaction_matrix_exp(p,l,slice,-1.)

h5write("tests/DetRatioCheck.h5", "V_real", real(V))
h5write("tests/DetRatioCheck.h5", "V_imag", imag(V))
h5write("tests/DetRatioCheck.h5", "Vminus_real", real(Vminus))
h5write("tests/DetRatioCheck.h5", "Vminus_imag", imag(Vminus))

hsfield = copy(p.hsfield)
newhsfield = copy(p.hsfield)
newhsfield[:,site,slice] = new_op[:]
p.hsfield[:] = newhsfield[:]
newV = interaction_matrix_exp(p,l,slice)

h5write("tests/DetRatioCheck.h5", "newV_real", real(newV))
h5write("tests/DetRatioCheck.h5", "newV_imag", imag(newV))

p.hsfield[:] = hsfield[:]

s.current_slice = slice
detratio = calc_detratio(s,p,l,site,new_op)
deltai = copy(s.delta_i)
M = copy(s.M)

h5write("tests/DetRatioCheck.h5", "detratio_real", real(detratio))
h5write("tests/DetRatioCheck.h5", "detratio_imag", imag(detratio))
h5write("tests/DetRatioCheck.h5", "deltai_real", real(deltai))
h5write("tests/DetRatioCheck.h5", "deltai_imag", imag(deltai))
h5write("tests/DetRatioCheck.h5", "M_real", real(M))
h5write("tests/DetRatioCheck.h5", "M_imag", imag(M))

h5write("tests/DetRatioCheck.h5", "greens_imag", imag(s.greens))
h5write("tests/DetRatioCheck.h5", "greens_real", real(s.greens))

first_term = /((s.greens - eye_full)[:,site:l.sites:end], s.M)
second_term = s.delta_i * s.greens[site:l.sites:end,:]
gupdated = s.greens + first_term * second_term

h5write("tests/DetRatioCheck.h5", "gupdated_imag", imag(gupdated))
h5write("tests/DetRatioCheck.h5", "gupdated_real", real(gupdated))